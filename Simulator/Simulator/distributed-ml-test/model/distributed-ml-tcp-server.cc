#include "ns3/distributed-ml-tcp-server.h"
#include "ns3/distributed-ml-utils.h"

#include "ns3/address.h"
#include "ns3/address-utils.h"
#include "ns3/log.h"
#include "ns3/inet-socket-address.h"
#include "ns3/inet6-socket-address.h"
#include "ns3/node.h"
#include "ns3/socket.h"
#include "ns3/udp-socket.h"
#include "ns3/simulator.h"
#include "ns3/socket-factory.h"
#include "ns3/packet.h"
#include "ns3/uinteger.h"
#include "ns3/trace-source-accessor.h"
#include "ns3/udp-socket-factory.h"

#include <iostream>
#include <vector>

namespace ns3{

NS_LOG_COMPONENT_DEFINE ("DistributedMlTcpServerApplication");

NS_OBJECT_ENSURE_REGISTERED (DistributedMlTcpServer);

TypeId
DistributedMlTcpServer::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::DistributedMlTcpServer")
    .SetParent<Application> ()
    .SetGroupName("Applications")
    .AddConstructor<DistributedMlTcpServer> ()
    .AddAttribute ("RemoteAddress",
                   "The local Address of the outbound packets",
                   AddressValue (),
                   MakeAddressAccessor (&DistributedMlTcpServer::m_remote),
                   MakeAddressChecker ())
    .AddAttribute ("RemotePort",
                   "The destination port of the outbound packets",
                   UintegerValue (0),
                   MakeUintegerAccessor (&DistributedMlTcpServer::m_port),
                   MakeUintegerChecker<uint16_t> ())
    .AddAttribute ("DataRate",
                   "The packets trans data rate",
                   DataRateValue (DataRate ("1Mbps")),
                   MakeDataRateAccessor (&DistributedMlTcpServer::m_dataRate),
                   MakeDataRateChecker ())
    .AddAttribute ("PacketSize",
                   "The size of packets",
                   UintegerValue (100),
                   MakeUintegerAccessor (&DistributedMlTcpServer::m_packetSize),
                   MakeUintegerChecker<uint32_t> ())
    .AddAttribute ("NumPackets",
                   "The number of packets the application will send",
                   UintegerValue (1),
                   MakeUintegerAccessor (&DistributedMlTcpServer::m_nPackets),
                   MakeUintegerChecker<uint32_t> ())
    .AddAttribute ("NumClients",
                   "The number of clients the server will connect",
                   UintegerValue (3),
                   MakeUintegerAccessor (&DistributedMlTcpServer::m_clients),
                   MakeUintegerChecker<uint32_t> ())
  ;
  return tid;
}

DistributedMlTcpServer::DistributedMlTcpServer ()
  : m_socket (0),
  m_sync(false),
  m_clients(3)
{
  m_header.SetSeq(1);
}

DistributedMlTcpServer::~DistributedMlTcpServer()
{
  m_socket = 0;
}


void
DistributedMlTcpServer::SetSocket (Ptr<Socket> socket)
{
  m_socket = socket;
}

void
DistributedMlTcpServer::SetAttributes (Address Address, Ptr<Socket> Socket, uint32_t PacketSize, uint32_t NumPackets, uint32_t NumClients, DataRate DataRate)
{
  m_remote = Address;
  m_socket = Socket;
  m_nPackets = NumPackets;
  m_dataRate = DataRate;
  m_packetSize = PacketSize;
  m_clients = NumClients;
}



void
DistributedMlTcpServer::SetTensor(uint64_t t_addr, uint32_t size)
{
  Buff = MlBuffer(t_addr, size);

}



/*TODO:: Currently, we're not able to use std::vector<uint64_t> as python input.
        To avoid this awkward situation, we use the std::vector<uint32_t> as input, while
        adjusting it in the python input.
*/
void
DistributedMlTcpServer::SetModel(std::vector<uint32_t> t_addrs, std::vector<uint32_t> sizes)
{
  
  uint64_t* addrs = new uint64_t[int(t_addrs.size()/2)];
  if (!t_addrs.empty())
  {
      memcpy(addrs, &t_addrs[0], t_addrs.size()*sizeof(uint32_t));
  }

  Buff = MlBuffer(addrs, sizes);
  
}


void 
DistributedMlTcpServer::SetTrigger(bool trigged){
  m_trigged = trigged;
}


bool
DistributedMlTcpServer::GetTrigger(void){
  return m_trigged;
}


void
DistributedMlTcpServer::Initialize(void){
  if (Buff.size()==0){
    Ptr<Packet> packet = Create<Packet> (100);
    Buff = MlBuffer(packet);
  }
  Buff.Zero();
  
  m_nPackets *= m_clients;

  if (m_socket->Bind (m_remote) == -1)
  {
    printf ("Failed to bind socket");
  }

  m_socket->Listen ();

}


void
DistributedMlTcpServer::StartApplication (void)
{
  Initialize();
  NS_LOG_INFO("SERVER:: number of required packets: " << m_nPackets);
  
  m_socket->SetAcceptCallback (
    MakeNullCallback<bool, Ptr<Socket>, const Address &> (),
    MakeCallback (&DistributedMlTcpServer::HandleAccept, this));
  m_socket->SetRecvCallback (MakeCallback (&DistributedMlTcpServer::HandleRead, this));

  // m_socket->SetSendCallback(MakeCallback(&DistributedMlTcpServer::HandleSend, this));
}


void 
DistributedMlTcpServer::StopApplication ()     // Called at time specified by Stop
{
  NS_LOG_INFO("Server:: When stop, server received Tensor size: " << Buff.size());
  while(!m_socketList.empty ()) //these are accepted sockets, close them
    {
      Ptr<Socket> acceptedSocket = m_socketList.front ();
      m_socketList.pop_front ();
      acceptedSocket->Close ();
    }
  if (m_socket) 
    {
      m_socket->Close ();
      m_socket->SetRecvCallback (MakeNullCallback<void, Ptr<Socket> > ());
    }
}


void 
DistributedMlTcpServer::HandleAccept (Ptr<Socket> s, const Address& from)
{
  NS_LOG_FUNCTION (this << s << from);
  s->SetRecvCallback (MakeCallback (&DistributedMlTcpServer::HandleRead, this));
  m_socketList.push_back (s);
  NS_LOG_INFO("Server:: number of accepted socket: " << m_socketList.size());
  
}


void 
DistributedMlTcpServer::HandleRead (Ptr<Socket> socket)
{
  do{
    NS_LOG_FUNCTION (this << socket);
    Ptr<Packet> packet;
    Address from;
    Address localAddress;
    if (packet = socket->RecvFrom (from))
    {
      if (packet->GetSize () == 0)
        { //EOF
          printf("end of sending. \n");
          break;
        }

      socket->GetSockName (localAddress);
      PacketReceived (socket, packet, from, localAddress);
    }
    else{
      SendPacket();
      return;
    }
  }while(m_nPackets>0);
}


void 
DistributedMlTcpServer::SendPacket(){
  
  if (GetTrigger()){
    SetTrigger(false);
    
    Buff.PasteToMem();
    
    Processing();   // Server Evaluation Function
    
    while(!memory_socketList.empty ()) //these are accepted sockets, close them
    {
      Ptr<Socket> socket = memory_socketList.front ();      
      
      Address from, to;
      socket->GetSockName (from);
      socket->GetPeerName (to);
      
      Buff.FedSend (socket, m_packetSize, m_dataRate);
      
      m_nPackets-=1;
      memory_socketList.pop_front();
    }
    Buff.Zero();   //Set Buff value to be zero, to FedAvg new data
    m_TV.Initialize();
  }

}


void
DistributedMlTcpServer::PacketReceived (Ptr<Socket> socket, const Ptr<Packet> &p, const Address &from,
                            const Address &localAddress)
{
  SeqTsSizeHeader header;
  Ptr<Packet> buffer;

  auto itBuffer = m_buffer.find (from);
  if (itBuffer == m_buffer.end ())
    {
      itBuffer = m_buffer.insert (std::make_pair (from, Create<Packet> (0))).first;
    }

  buffer = itBuffer->second;

  buffer->AddAtEnd (p);
  buffer->PeekHeader (header);

  NS_ABORT_IF (header.GetSize () == 0);


  while (buffer->GetSize () >= header.GetSize ())
    { 
      // NS_LOG_INFO("Server::  Removing packet of size " << header.GetSize () << " from buffer of size " << buffer->GetSize ());

      Ptr<Packet> complete = buffer->CreateFragment (0, static_cast<uint32_t> (header.GetSize ()));
      buffer->RemoveAtStart (static_cast<uint32_t> (header.GetSize ()));

      uint32_t seq = Buff.FedAvg(complete);
      
      if(seq==0){
        m_count += 1;
        memory_socketList.push_back(socket);
        //TODO: the synchronous and asynchronous algorithm depends on When to set m_trigged=true 
        if(TriggerLogic()){
          SetTrigger(true);
        }
        seq = 1;
      }


      if (buffer->GetSize () > header.GetSerializedSize ())
        {
          buffer->PeekHeader (header);
        }
      else
        {
          break;
        }
      
    }
}


bool DistributedMlTcpServer::TriggerLogic(){
  if (m_TV.get("count")>=3){
    return true;
  }else{
    return false;
  }


}


}