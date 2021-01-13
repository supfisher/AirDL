import ns.distributedml as dml
import ns.core
import ns.internet
import ns.network
import ns.point_to_point
import ns.csma
import ns.wifi

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import torch.optim as optim

from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

import gc
"""
    Build Neural network
"""

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output



class Model:
	def __init__(self, *args, **kwargs):
		train_kwargs = {'batch_size': 128}
		transform=transforms.Compose([
		        transforms.ToTensor(),
		        transforms.Normalize((0.1307,), (0.3081,))
		    ])
		dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
		self.train_loader = torch.utils.data.DataLoader(dataset, **train_kwargs)

		self.model = Net()

		self.optimizer = optim.Adadelta(self.model.parameters(), lr=1)

	def load_state(self, weight):
		self.model.load_state_dict(weight)

	def get_state(self):
		return self.model.state_dict()
	# Currently, we're not able to use std::vector<uint64_t> as python input.
	# To avoid this awkward thing, we use the std::vector<uint32_t> as input, while
	# adjusting it in the python input.
	def long2int(self, longlist):
		intlist = []    
		for l in longlist:
			intlist.append(l%(2**32))
			intlist.append(l//(2**32))
		return intlist

	def addrs(self):
		addr_list = []
		sizes = []
		for params in self.model.parameters():
			addr_list.append(params.data.data_ptr())
			sizes.append(params.data.view(-1).shape[0])

		return self.long2int(addr_list), sizes

	def train(self):

		self.model.train()

		for batch_idx, (data, target) in enumerate(self.train_loader):
			self.optimizer.zero_grad()
			output = self.model(data)

			loss = F.nll_loss(output, target)
			loss.backward()
			self.optimizer.step()
			if batch_idx % 100 == 0:
				print('TRAIN: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
			            0, batch_idx * len(data), len(self.train_loader.dataset),
			            100. * batch_idx / len(self.train_loader), loss.item()))	
				if batch_idx == 200:
					break	


	def evaluate(self):
		self.model.eval()
		for batch_idx, (data, target) in enumerate(self.train_loader):
			output = self.model(data)
			loss = F.nll_loss(output, target)
			if batch_idx % 100 == 0:
				print('EVAL: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				                0, batch_idx * len(data), len(self.train_loader.dataset),
				                100. * batch_idx / len(self.train_loader), loss.item()))
				if batch_idx == 200:
					break


def main_work():
    train_kwargs = {'batch_size': 128}
    transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset, **train_kwargs)

    model = Net()

    optimizer = optim.Adadelta(model.parameters(), lr=1)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

    return model, train_loader, optimizer, scheduler



# // Default Network Topology
# //
# //       10.1.1.0
# // n0 -------------- n1   n2   n3   n4
# //    point-to-point  |    |    |    |
# //                    ================
# //                      LAN 10.1.2.0




class PyDistributedMlTcpClient(dml.DistributedMlTcpClient):
    # def __init__(self, address, model, train_loader, optimizer, scheduler, *args, **kwargs):
    def __init__(self, address, model, *args, **kwargs):
        super(PyDistributedMlTcpClient, self).__init__(*args, **kwargs)
        self.model = model
        self.address = address

        self.id = kwargs['ID']

    # def Processing(self):
    # 	print("PYTHON:: Client Processing...")
    #     # self.model.train()


class ClientHelperSon(dml.DistributedMlTcpClientHelper):
    def __init__(self, address, models):
        self.address = address
        super(ClientHelperSon, self).__init__(address)
        self.models = models
        self.apps = ns.network.ApplicationContainer()

    def Install(self, nodes):
    	for i in range(len(self.models)):
    		app = self.InstallPriv(nodes.Get(i), self.models[i], i)
    		self.apps.Add(app)
    	return self.apps


    def InstallPriv(self, node, model, id):
        socket = self.CreateSocket (node)
        
        app = PyDistributedMlTcpClient(self.address, model, ID=id)
        app.SetAttributes(Address=self.address, Socket=socket, NumPackets=3)

        data_addr, sizes = model.addrs()
        
        app.SetModel(data_addr, sizes)

        app.SetTrigger(True)
        node.AddApplication (app)
        return app


class PyDistributedMlTcpServer(dml.DistributedMlTcpServer):
	def __init__(self, model, *args, **kwargs):
		super(PyDistributedMlTcpServer, self).__init__(*args, **kwargs)
		self.model = model

	def TriggerLogic(self):
		print("self.m_TV.get('count'):  ", self.m_TV.get("count"))
		if (self.m_TV.get("count")>=2):
			return True
		else:
			return False

	# def Processing(self):
	# 	print("PYTHON:: Server Processing...")
	# 	# self.model.evaluate()




class ServerHelperSon(dml.DistributedMlTcpServerHelper):
	def __init__(self, address, model):
		self.address = address
		super(ServerHelperSon, self).__init__(address)
		self.apps = ns.network.ApplicationContainer()
		self.model = model


	def InstallPriv(self, node):      
		serverSocket = self.CreateSocket (node)
		app = PyDistributedMlTcpServer(self.model)
		app.SetAttributes(Address=self.address, Socket=serverSocket, NumPackets=3, NumClients=2)
		data_addr, sizes = self.model.addrs()
		app.SetModel(data_addr, sizes)
		node.AddApplication (app)
		return app



# // Default Network Topology
# //
# //   Wifi 10.1.3.0						systemCsma
# //                 AP
# //  *    *    *    *				
# //  |    |    |    |    10.1.1.0	server
# // n5   n6   n7   n0 -------------- n1   n2   n3   n4
# //                   point-to-point  |    |    |    |
# //                                   ================
# //    systemWifi                       LAN 10.1.2.0



Enable_MPI = True
if __name__ == "__main__":
	server_model = Model()

	if Enable_MPI:
		m = dml.MpiHelper()
		print("sys.argv: ", sys.argv)
		m.Enable(sys.argv)
		systemId = m.GetSystemId ()
		systemCount = m.GetSize ()
		print("systemId: ", systemId)
		print("systemCount: ", systemCount)

		systemWifi = 0
		systemCsma = 1
	else:
		systemId = 0
		systemWifi = 0
		systemCsma = 0



	cmd = ns.core.CommandLine()
	cmd.nCsma = 1
	cmd.verbose = "True"
	cmd.nWifi = 1
	cmd.tracing = "False"

	cmd.AddValue("nCsma", "Number of \"extra\" CSMA nodes/devices")
	cmd.AddValue("nWifi", "Number of wifi STA devices")
	cmd.AddValue("verbose", "Tell echo applications to log if true")
	cmd.AddValue("tracing", "Enable pcap tracing")

	cmd.Parse(sys.argv)

	nCsma = int(cmd.nCsma)
	verbose = cmd.verbose
	nWifi = int(cmd.nWifi)
	tracing = cmd.tracing

	# The underlying restriction of 18 is due to the grid position
	# allocator's configuration; the grid layout will exceed the
	# bounding box if more than 18 nodes are provided.
	if nWifi > 18:
		print ("nWifi should be 18 or less; otherwise grid layout exceeds the bounding box")
		sys.exit(1)

	if verbose == "True":
		ns.core.LogComponentEnable("DistributedMlTcpClientApplication", ns.core.LOG_LEVEL_INFO)
		ns.core.LogComponentEnable("DistributedMlTcpServerApplication", ns.core.LOG_LEVEL_INFO)
		ns.core.LogComponentEnable("DistributedMlUtils", ns.core.LOG_LEVEL_INFO)
		
	p2pNodes = ns.network.NodeContainer()
	node0 = ns.network.Node(systemWifi)
	node1 = ns.network.Node(systemCsma)
	p2pNodes.Add(node0)
	p2pNodes.Add(node1)

	pointToPoint = ns.point_to_point.PointToPointHelper()
	pointToPoint.SetDeviceAttribute("DataRate", ns.core.StringValue("5Mbps"))
	pointToPoint.SetChannelAttribute("Delay", ns.core.StringValue("2ms"))

	p2pDevices = pointToPoint.Install(p2pNodes)

	csmaNodes = ns.network.NodeContainer()
	csmaNodes.Create(nCsma, systemCsma)
	csmaNodes.Add(p2pNodes.Get(1))
	

	csma = ns.csma.CsmaHelper()
	csma.SetChannelAttribute("DataRate", ns.core.StringValue("100Mbps"))
	csma.SetChannelAttribute("Delay", ns.core.TimeValue(ns.core.NanoSeconds(6560)))

	csmaDevices = csma.Install(csmaNodes)

	wifiStaNodes = ns.network.NodeContainer()
	wifiStaNodes.Create(nWifi, systemWifi)
	wifiApNode = p2pNodes.Get(0)

	channel = ns.wifi.YansWifiChannelHelper.Default()
	phy = ns.wifi.YansWifiPhyHelper.Default()
	phy.SetChannel(channel.Create())

	wifi = ns.wifi.WifiHelper()
	wifi.SetRemoteStationManager("ns3::AarfWifiManager")

	mac = ns.wifi.WifiMacHelper()
	ssid = ns.wifi.Ssid ("ns-3-ssid")

	mac.SetType ("ns3::StaWifiMac", "Ssid", ns.wifi.SsidValue(ssid), "ActiveProbing", ns.core.BooleanValue(False))
	staDevices = wifi.Install(phy, mac, wifiStaNodes)

	mac.SetType("ns3::ApWifiMac","Ssid", ns.wifi.SsidValue (ssid))
	apDevices = wifi.Install(phy, mac, wifiApNode)

	mobility = ns.mobility.MobilityHelper()
	mobility.SetPositionAllocator ("ns3::GridPositionAllocator", "MinX", ns.core.DoubleValue(0.0),
									"MinY", ns.core.DoubleValue (0.0), "DeltaX", ns.core.DoubleValue(5.0), "DeltaY", ns.core.DoubleValue(10.0),
	                                 "GridWidth", ns.core.UintegerValue(3), "LayoutType", ns.core.StringValue("RowFirst"))

	mobility.SetMobilityModel ("ns3::RandomWalk2dMobilityModel", "Bounds", ns.mobility.RectangleValue(ns.mobility.Rectangle (-50, 50, -50, 50)))
	mobility.Install(wifiStaNodes)

	mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel")
	mobility.Install(wifiApNode)

	stack = ns.internet.InternetStackHelper()
	stack.Install(csmaNodes)
	stack.Install(wifiApNode)
	stack.Install(wifiStaNodes)

	address = ns.internet.Ipv4AddressHelper()
	address.SetBase(ns.network.Ipv4Address("10.1.1.0"), ns.network.Ipv4Mask("255.255.255.0"))
	p2pInterfaces = address.Assign(p2pDevices)

	address.SetBase(ns.network.Ipv4Address("10.1.2.0"), ns.network.Ipv4Mask("255.255.255.0"))
	csmaInterfaces = address.Assign(csmaDevices)

	address.SetBase(ns.network.Ipv4Address("10.1.3.0"), ns.network.Ipv4Mask("255.255.255.0"))
	address.Assign(staDevices)
	address.Assign(apDevices)


	ns.internet.Ipv4GlobalRoutingHelper.PopulateRoutingTables()

	sinkPort = 8080
	sinkAddress = (ns.network.InetSocketAddress (csmaInterfaces.GetAddress(nCsma), sinkPort))


	if(systemId==systemCsma):
		serverHelper = ServerHelperSon(ns.network.InetSocketAddress (ns.network.Ipv4Address.GetAny (), sinkPort), server_model)

		serverHelper.SetAttribute("NumPackets", ns.core.UintegerValue(3))
		serverHelper.SetAttribute("PacketSize", ns.core.UintegerValue(1000))		

		serverApp = serverHelper.Install(csmaNodes.Get(nCsma))

		serverApp.Start(ns.core.Seconds(1.0))
		serverApp.Stop(ns.core.Seconds(20000.0))

	if systemId==systemWifi:
		client_models = []
		for i in range(nWifi+nCsma):
			client_model = Model()
			client_model.load_state(server_model.get_state())
			client_models.append(client_model)

		# clientHelper = ClientHelperSon(sinkAddress, client_models[0:nWifi])
		# clientApp1 = clientHelper.Install(wifiStaNodes)
		# clientApp1.Start(ns.core.Seconds(1.0))
		# clientApp1.Stop(ns.core.Seconds(20000.0))

		clientHelper = ClientHelperSon(sinkAddress, client_models[nWifi:])
		clientApp2 = clientHelper.Install(csmaNodes)
		clientApp2.Start(ns.core.Seconds(1.0))
		clientApp2.Stop(ns.core.Seconds(20000.0))

	
	ns.core.Simulator.Stop(ns.core.Seconds(20000.0))

	ns.core.Simulator.Run()
	ns.core.Simulator.Destroy()


	if Enable_MPI:
		m.Disable()







