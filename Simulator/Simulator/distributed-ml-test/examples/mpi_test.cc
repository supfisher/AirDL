/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation;
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 */

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/csma-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/distributedml-module.h"
#include "ns3/ipv4-global-routing-helper.h"
#include "ns3/mpi-module.h"


#include <fstream>
#include <iostream>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("ThirdScriptExample");

// ===========================================================================
//
//         node 0                 node 1
//   +----------------+    +----------------+
//   |    ns-3 TCP    |    |    ns-3 TCP    |
//   +----------------+    +----------------+
//   |    10.1.1.1    |    |    10.1.1.2    |
//   +----------------+    +----------------+
//   | point-to-point |    | point-to-point |
//   +----------------+    +----------------+
//           |                     |
//           +---------------------+
//                5 Mbps, 2 ms
//
//
// We want to look at changes in the ns-3 TCP congestion window.  We need
// to crank up a flow and hook the CongestionWindow attribute on the socket
// of the sender.  Normally one would use an on-off application to generate a
// flow, but this has a couple of problems.  First, the socket of the on-off 
// application is not created until Application Start time, so we wouldn't be 
// able to hook the socket (now) at configuration time.  Second, even if we 
// could arrange a call after start time, the socket is not public so we 
// couldn't get at it.
//
// So, we can cook up a simple version of the on-off application that does what
// we want.  On the plus side we don't need all of the complexity of the on-off
// application.  On the minus side, we don't have a helper, so we have to get
// a little more involved in the details, but this is trivial.
//
// So first, we create a socket and do the trace connect on it; then we pass 
// this socket into the constructor of our simple application which we then 
// install in the source node.
// ===========================================================================
//
class ClientSon: public DistributedMlTcpClient
{
public:
  static TypeId GetTypeId (void);

  ClientSon();

  virtual ~ClientSon ();

  void Processing(void);
};



TypeId
ClientSon::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::ClientSon")
    .SetParent<DistributedMlTcpClient> ()
    .AddConstructor<ClientSon> ()
    ;
  return tid;
}




ClientSon::ClientSon(): DistributedMlTcpClient(){
  printf("Calling ClientSon!!\n");
}


ClientSon::~ClientSon(){

}


void ClientSon::Processing(void){
  printf("WTF!!  It works in Cpp\n");
  // This->SetTrigger(true);
}



template <class T>
class ClientHelperSon: public DistributedMlTcpClientHelper
{
public:
  ClientHelperSon<T> (Address ip, uint16_t port): DistributedMlTcpClientHelper(ip, port){
    m_factory.SetTypeId (T::GetTypeId());
    std::cout<< "typeid:: " << T::GetTypeId() << std::endl;
  };
  ClientHelperSon<T> (Address ip): DistributedMlTcpClientHelper(ip){
    m_factory.SetTypeId (T::GetTypeId());
    std::cout<< "typeid:: " << T::GetTypeId() << std::endl;
  };

  ApplicationContainer Install (NodeContainer c) const;
  Ptr<DistributedMlTcpClient> InstallPriv (Ptr<Node> node) const;
  ObjectFactory m_factory = GetObjectFactory();
};


template <class T>
ApplicationContainer
ClientHelperSon<T>::Install (NodeContainer c) const
{
  ApplicationContainer apps;
  
  for (NodeContainer::Iterator i = c.Begin (); i != c.End (); ++i)
    {
      apps.Add (InstallPriv (*i));
    }

  return apps;
}


template <class T>
Ptr<DistributedMlTcpClient>
ClientHelperSon<T>::InstallPriv (Ptr<Node> node) const
{
  Ptr<Socket> socket = CreateSocket (node);
  Ptr<T> app = m_factory.Create<T> ();
  app->SetSocket (socket);
  node->AddApplication (app);
  return app;
}


int 
main (int argc, char *argv[])
{

  
  bool verbose = true;
  uint32_t nCsma = 2;

  uint32_t nPackets = 3;

  CommandLine cmd (__FILE__);
  cmd.AddValue ("nCsma", "Number of \"extra\" CSMA nodes/devices", nCsma);
  cmd.AddValue ("verbose", "Tell echo applications to log if true", verbose);

  cmd.Parse (argc,argv);

  LogComponentEnable ("DistributedMlTcpClientApplication", LOG_LEVEL_INFO);
  LogComponentEnable ("DistributedMlTcpServerApplication", LOG_LEVEL_INFO);


  // Enable parallel simulator with the command line arguments
  MpiInterface::Enable (&argc, &argv);


  uint32_t systemId = MpiInterface::GetSystemId ();
  uint32_t systemCount = MpiInterface::GetSize ();

  // System id of Wifi side
  uint32_t systemWifi = 0;

  // System id of CSMA side
  uint32_t systemCsma = systemCount - 1;


  nCsma = nCsma == 0 ? 1 : nCsma;

  NodeContainer p2pNodes;
  Ptr<Node> p2pNode1 = CreateObject<Node> (systemWifi);
  Ptr<Node> p2pNode2 = CreateObject<Node> (systemCsma);
  p2pNodes.Add (p2pNode1);
  p2pNodes.Add (p2pNode2);

  NodeContainer csmaNodes;
  csmaNodes.Add (p2pNodes.Get (1));
  csmaNodes.Create (nCsma, systemCsma);

  PointToPointHelper pointToPoint;
  pointToPoint.SetDeviceAttribute ("DataRate", StringValue ("5Mbps"));
  pointToPoint.SetChannelAttribute ("Delay", StringValue ("2ms"));

  NetDeviceContainer p2pDevices;
  p2pDevices = pointToPoint.Install (p2pNodes);

  CsmaHelper csma;
  csma.SetChannelAttribute ("DataRate", StringValue ("100Mbps"));
  csma.SetChannelAttribute ("Delay", TimeValue (NanoSeconds (6560)));

  NetDeviceContainer csmaDevices;
  csmaDevices = csma.Install (csmaNodes);

  InternetStackHelper stack;
  stack.Install (p2pNodes.Get (0));
  stack.Install (csmaNodes);

  Ipv4AddressHelper address;
  address.SetBase ("10.1.1.0", "255.255.255.0");
  Ipv4InterfaceContainer p2pInterfaces;
  p2pInterfaces = address.Assign (p2pDevices);

  address.SetBase ("10.1.2.0", "255.255.255.0");
  Ipv4InterfaceContainer csmaInterfaces;
  csmaInterfaces = address.Assign (csmaDevices);

  Ipv4GlobalRoutingHelper::PopulateRoutingTables ();


  uint16_t sinkPort = 8080;
  Address sinkAddress (InetSocketAddress (p2pInterfaces.GetAddress (0), sinkPort));


  if (systemId == systemWifi){
    DistributedMlTcpServerHelper serverHelper (InetSocketAddress (Ipv4Address::GetAny (), sinkPort));
    serverHelper.SetAttribute("NumPackets", UintegerValue(nPackets));
    serverHelper.SetAttribute("PacketSize", UintegerValue(1000));
    serverHelper.SetAttribute("NumClients", UintegerValue(nCsma+1));

    ApplicationContainer serverApp = serverHelper.Install(p2pNodes.Get (0));
    float serverVec[] = {1000.0, 2.0, 3.4, 7.0, 50.0};
    serverHelper.SetTensor(serverApp.Get(0), (uint64_t)&serverVec, 5);
    serverApp.Start (Seconds (1.0));
    serverApp.Stop (Seconds (20000.0));
  }
  if (systemId == systemCsma){
    ClientHelperSon<ClientSon> clientHelper(sinkAddress);

    clientHelper.SetAttribute("NumPackets", UintegerValue(nPackets));
    clientHelper.SetAttribute("PacketSize", UintegerValue(1000));

    ApplicationContainer clientApp = clientHelper.Install(csmaNodes);
    
    float clientVec0[] = {1000.0, 2.0, 3.4, 7.0, 50.0};
    float clientVec1[] = {1000.0, 2.0, 3.4, 7.0, 50.0};
    float clientVec2[] = {1000.0, 2.0, 3.4, 7.0, 50.0};

    printf("Scratch.cc::  addr: %lu\n", (uint64_t)&clientVec0);
    clientHelper.SetTensor(clientApp.Get(0), (uint64_t)&clientVec0, 5);

    printf("Scratch.cc::  addr: %lu\n", (uint64_t)&clientVec1);
    clientHelper.SetTensor(clientApp.Get(1), (uint64_t)&clientVec1, 5);

    printf("Scratch.cc::  addr: %lu\n", (uint64_t)&clientVec2);
    clientHelper.SetTensor(clientApp.Get(2), (uint64_t)&clientVec2, 5);


    clientHelper.SetTrigger(clientApp, true);
    clientApp.Start (Seconds (1.0));
    clientApp.Stop (Seconds (20000.0)); 

    }
  Simulator::Run ();
  Simulator::Destroy ();

  MpiInterface::Disable ();
  return 0;
}

