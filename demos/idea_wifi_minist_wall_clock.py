"""
	A simple example to run the FedAvg algorithm.
	It concludes three clients and one server. The server received model from three clients, and 
	do the FedAvg algorithm. After the averaging, the server sends packets back to all clients.
	And these client-server rounds happen on three times.
"""
import sys
import os
import argparse
import ns.distributedml as dml
import ns.network
from minist import AirTask
from helper import ClientHelperSon, ServerHelperSon
from wifi import WifiCell
from wifi import P2PChannel
from wifi import Network
from wifi import build_n_wifi_cells
# from wifi import MobilityRandomWalk2d, WifiYansChannel
from node import Agent
import ns.wifi
import ns.internet
import ns.mobility
import ns.point_to_point

from base import Tracer, Mpi

import torchvision
import torch
from torchvision import transforms

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", default=3, type=int,
                        help="number of epochs.")
parser.add_argument("--nCells", default=1, type=int,
                        help="number of cells.")
parser.add_argument("--nWifiPerCell", default=4, type=int,
                        help="number of wifi nodes in one cell.")
parser.add_argument("--packet_size", default=1024, type=int,
                        help="packet_size.")
parser.add_argument("--batch_size", default=128, type=int,
                        help="batch_size.")
parser.add_argument("--local_epochs", default=1, type=int,
                        help="number of local epochs.")
parser.add_argument("--error_rate", default=0, type=float,
                        help="error rate for receiver.")
parser.add_argument("--nActivePerCell", default=4, type=int,
                        help="the number activated agents per cell.")
parser.add_argument("--sleeping_time", default=0, type=float,
                        help="sleeping time.")
parser.add_argument("--noise_ratio", default=0, type=float,
                        help="noise ratio, default is 0.")
parser.add_argument("--noise_type", default="add", type=str,
                        help="noise type, default is add-noise.")
parser.add_argument("--verbose", action='store_true',
                        help="if added, enable log.")
parser.add_argument("--mpi", action='store_true',
                        help="if added, enable mpi.")
parser.add_argument("--tracing", action='store_true',
                        help="if added, enable tracing.")
parser.add_argument("--saved_dir", default='saved_minist', type=str,
                        help="number of local epochs.")

parser.add_argument("--part_ratio", default='1,1,1,1', type=str,
						help='data partition ratio, here we only consider 4 agents')



args = parser.parse_args()


# // Default Network Topology
# //
# //   Wifi 10.1.3.0									
# //                 AP
# //  *    *    *    *									Wifi 10.1.4.0
# //  |    |    |    |    10.1.1.0		    10.1.2.0
# // n5   n6   n7   n8 -------------- n0 --------------	n1   n2   n3   n4
# //                   point-to-point  	 point-to-point |    |    |    |
# //                                   					*    *    *    * 
# //                                     				Ap


if __name__ == "__main__":
	if args.verbose:
		ns.core.LogComponentEnable("DistributedMlTcpAgentApplication", ns.core.LOG_LEVEL_INFO)
		ns.core.LogComponentEnable("DistributedMlUtils", ns.core.LOG_LEVEL_INFO)

	# print("sys.argv: ", sys.argv)  ## This command is necessary!! WTF
	Mpi.enable(sys.argv)
	# print("enabled mpi...")
	systemId = Mpi.rank
	systemCount = Mpi.world_size
	
	nCells = max(args.nCells, systemCount-1)
	BackBoneNet, Cells = build_n_wifi_cells(nCells, args.nWifiPerCell)


	end_time = args.epochs*1000

	systemServer = 0
	systemWifi = list(range(1, systemCount))*(nCells//(systemCount-1))
	systemWifi.extend([1]*(nCells-len(systemWifi)))

	

	net = Network(systemWifi=systemWifi, systemServer=systemServer, BackBoneNet=BackBoneNet, Cells=Cells, errorRate=args.error_rate)

	server_agent, wifi_cells, sinkPort, sinkAddress = net.config()

	args.saved_dir = os.path.join(args.saved_dir, "outputs-"+str(32))

	record_prefix = "-systemCount-"+str(systemCount-1) + "-nCells-"+str(nCells)
	
	nTotalAgents = sum([len(wifi_cell) for wifi_cell in wifi_cells])		
	nActiveAgents = args.nActivePerCell*len(wifi_cells)
	active_ratio = args.nActivePerCell/args.nWifiPerCell

	if(systemId==systemServer):
		print("nActiveAgents: ", nActiveAgents, "nTotalAgents: ", nTotalAgents)
		print("systemWifi: ", systemWifi)

		server_task = AirTask(global_rank=-1, global_size=nTotalAgents, log=os.path.join(args.saved_dir, "tf"+record_prefix))

		serverHelper = ServerHelperSon(ns.network.InetSocketAddress (ns.network.Ipv4Address.GetAny (), sinkPort), server_task, num_packets=args.epochs, num_clients=nActiveAgents, packet_size=args.packet_size)

		App = serverHelper.Install(server_agent.nodes)
		App.Start(ns.core.Seconds(1.0))
		App.Stop(ns.core.Seconds(end_time))


	global_rank = (systemId-1)*args.nActivePerCell

	for i, wifi_cell in enumerate(wifi_cells):
		if systemId == systemWifi[i]:

			kwargs = {\
				"local_epochs": args.local_epochs,\
				"batch_size": args.batch_size,\
				"active_ratio": active_ratio,\
				"sleeping_time": args.sleeping_time,\
				"noise_ratio": args.noise_ratio, \
				"noise_type": args.noise_type, \
				"part_ratio": [int(_) for _ in args.part_ratio.split(',')]
			}

			client_task = [AirTask(global_rank=global_rank+i, global_size=nTotalAgents, **kwargs) for i in range(len(wifi_cell))]
			
			clientHelper = ClientHelperSon(sinkAddress, client_task, num_packets=args.epochs, num_clients=nActiveAgents, packet_size=args.packet_size, energy_models=wifi_cell.staMlEnergyModels)
			App = clientHelper.Install(wifi_cell.sta.nodes)
			App.Start(ns.core.Seconds(1.0))
			App.Stop(ns.core.Seconds(end_time))

			# for i, node in enumerate(wifi_cell.sta):
			# 	tracer = Tracer(prefix=os.path.join(args.saved_dir, "trace"+record_prefix))

			# 	tracer.trace_ml_energy_consumation(wifi_cell.staMlEnergyModels[i], "ml_energy%s.txt"%(global_rank+i+1))
			# 	tracer.trace_wifi_energy_consumation(wifi_cell.staWifiEnergyModels[i], "energy%s.txt"%(global_rank+i+1))
				
			# 	tracer.trace_mobility(node, "mobility%s.txt"%(global_rank+i+1))
			# 	tracer.trace_cwnd(clientHelper[i].GetSocket(), "cwnd%s.txt"%(global_rank+i+1))
			# 	tracer.trace_drop(wifi_cell.staDevices.Get(i), "drop%s.pcap"%(global_rank+i+1))

			global_rank += args.nActivePerCell
			

	
	ns.core.Simulator.Stop(ns.core.Seconds(end_time))
	ns.core.Simulator.Run()
	ns.core.Simulator.Destroy()

	if args.mpi and systemId==args.systemServer:
		print("Disable mpi at final step, current time: ", dml.PyTimer.now("s"))
		Mpi.disable()
		

	

	




