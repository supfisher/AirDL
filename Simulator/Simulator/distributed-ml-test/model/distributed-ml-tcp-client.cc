#include "ns3/distributed-ml-tcp-client.h"
#include "ns3/distributed-ml-utils.h"

#include "ns3/log.h"
#include "ns3/ipv4-address.h"
#include "ns3/ipv6-address.h"
#include "ns3/nstime.h"
#include "ns3/inet-socket-address.h"
#include "ns3/inet6-socket-address.h"
#include "ns3/socket.h"
#include "ns3/simulator.h"
#include "ns3/socket-factory.h"
#include "ns3/packet.h"
#include "ns3/uinteger.h"
#include "ns3/trace-source-accessor.h"
#include "ns3/boolean.h"



namespace ns3{

NS_LOG_COMPONENT_DEFINE ("DistributedMlTcpClientApplication");

NS_OBJECT_ENSURE_REGISTERED (DistributedMlTcpClient);

TypeId
DistributedMlTcpClient::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::DistributedMlTcpClient")
    .SetParent<Application> ()
    .SetGroupName("Applications")
    .AddConstructor<DistributedMlTcpClient> ()
    .AddAttribute ("NumPackets",
                   "The number of packets the application will send",
                   UintegerValue (3),
                   MakeUintegerAccessor (&DistributedMlTcpClient::m_nPackets),
                   MakeUintegerChecker<uint32_t> ())
    .AddAttribute ("PacketSize",
                   "The size of packets",
                   UintegerValue (516),
                   MakeUintegerAccessor (&DistributedMlTcpClient::m_packetSize),
                   MakeUintegerChecker<uint32_t> ())
    .AddAttribute ("RemoteAddress",
                   "The destination Address of the outbound packets",
                   AddressValue (),
                   MakeAddressAccessor (&DistributedMlTcpClient::m_peer),
                   MakeAddressChecker ())
    .AddAttribute ("RemotePort",
                   "The destination port of the outbound packets",
                   UintegerValue (0),
                   MakeUintegerAccessor (&DistributedMlTcpClient::m_port),
                   MakeUintegerChecker<uint16_t> ())
    .AddAttribute ("DataRate",
                   "The packets trans data rate",
                   DataRateValue (DataRate ("1Mbps")),
                   MakeDataRateAccessor (&DistributedMlTcpClient::m_dataRate),
                   MakeDataRateChecker ())
  ;
  return tid;
}


DistributedMlTcpClient::DistributedMlTcpClient ()
  : m_socket (0), 
    m_peer (), 
    m_packetSize (516), 
    m_nPackets (3), 
    m_dataRate (DataRate ("1Mbps")), 
    m_seq(0),
    m_id(0)
{
  m_header.SetSeq(1);
}

DistributedMlTcpClient::DistributedMlTcpClient (uint32_t ID, uint32_t PacketSize, uint32_t NumPackets, DataRate DataRate)
  : m_packetSize (PacketSize), 
    m_nPackets (NumPackets), 
    m_dataRate (DataRate), 
    m_seq(0),
    m_id(ID)
{
  m_header.SetSeq(1);
}


DistributedMlTcpClient::~DistributedMlTcpClient()
{
  m_socket = 0;
}


void
DistributedMlTcpClient::SetSocket (Ptr<Socket> socket)
{
  m_socket = socket;
}

void
DistributedMlTcpClient::SetAttributes (Address Address, Ptr<Socket> Socket, uint32_t PacketSize, uint32_t NumPackets, DataRate DataRate)
{
  m_peer = Address;
  m_socket = Socket;
  m_nPackets = NumPackets;
  m_dataRate = DataRate;
  m_packetSize = PacketSize;
  NS_LOG_INFO("Client:: Socket connect m_peer!!" << m_peer);
}


void
DistributedMlTcpClient::SetTensor(uint64_t t_addr, uint32_t size)
{
  Buff = MlBuffer(t_addr, size);

}



/*TODO:: Currently, we're not able to use std::vector<uint64_t> as python input.
        To avoid this awkward situation, we use the std::vector<uint32_t> as input, while
        adjusting it in the python input.
*/
void
DistributedMlTcpClient::SetModel(std::vector<uint32_t> t_addrs, std::vector<uint32_t> sizes)
{
  
  uint64_t* addrs = new uint64_t[int(t_addrs.size()/2)];
  if (!t_addrs.empty())
  {
      memcpy(addrs, &t_addrs[0], t_addrs.size()*sizeof(uint32_t));
  }

  Buff = MlBuffer(addrs, sizes);
}


void 
DistributedMlTcpClient::SetTrigger(bool trigged){
  m_trigged = trigged;
}


bool
DistributedMlTcpClient::GetTrigger(void){
  return m_trigged;
}




void 
DistributedMlTcpClient::Initialize(void){
  if (Buff.size()==0){
    Ptr<Packet> packet = Create<Packet> (100);
    Buff = MlBuffer(packet);
  }

  m_socket->Bind ();
  m_socket->Connect (m_peer);
  
}


void
DistributedMlTcpClient::StartApplication (void)
{
  Initialize();
  NS_LOG_INFO("CLIENT:: number of required packets: " << m_nPackets);
  
  m_socket->SetSendCallback(MakeCallback(&DistributedMlTcpClient::HandleSend, this));

  m_socket->SetRecvCallback (MakeCallback (&DistributedMlTcpClient::HandleRead, this));

}

void 
DistributedMlTcpClient::StopApplication (void)
{
  NS_LOG_INFO("CLIENT:: when stop app client received buffer size: " << Buff.size());
  if (m_sendEvent.IsRunning ())
    {
      Simulator::Cancel (m_sendEvent);
    }

  if (m_socket)
    {
      m_socket->Close ();
    }
}


void DistributedMlTcpClient::HandleSend(Ptr<Socket> socket, uint32_t availableBufferSize){
  
  while(m_nPackets>0){
    if (m_seq==0 && GetTrigger()){
      m_seq = 1;
      SetTrigger(false);

      Buff.PasteToMem();  //TODO: PasteToMem has something wrong

      Processing();

      Buff.CopyFromMem();

      Address from, to;
      socket->GetSockName (from);
      socket->GetPeerName (to);

      Buff.FedSend (socket, m_packetSize, m_dataRate);
      m_nPackets-=1;

    }else{
      return;
    }
  }
}




void
DistributedMlTcpClient::HandleRead (Ptr<Socket> socket)
{
  NS_LOG_FUNCTION (this << socket);

  do{
    Ptr<Packet> packet;
    Address from;
    Address localAddress;
    if ((packet = socket->RecvFrom (from)))
      {
        if (packet->GetSize () == 0)
          { //EOF
            printf("end of sending. \n");
            break;
          }
        socket->GetSockName (localAddress);
        PacketReceived (packet, from, localAddress);
      }
    else{
      HandleSend(socket, 0);
      return;
    }
  }while(m_nPackets>0);
}



void
DistributedMlTcpClient::PacketReceived (const Ptr<Packet> &p, const Address &from,
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

  while (buffer->GetSize () >= header.GetSize () && header.GetSize () > 20)
    { 

      Ptr<Packet> complete = buffer->CreateFragment (0, static_cast<uint32_t> (header.GetSize ()));
      buffer->RemoveAtStart (static_cast<uint32_t> (header.GetSize ()));

      m_seq = Buff.FedUpdate(complete);

      if (m_seq==0)
      {      
        SetTrigger(true);
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


}