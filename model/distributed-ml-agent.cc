#include "ns3/distributed-ml-agent.h"
#include "ns3/distributed-ml-agent.h"
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

NS_LOG_COMPONENT_DEFINE ("DistributedMlTcpAgentApplication");

NS_OBJECT_ENSURE_REGISTERED (DistributedMlTcpAgent);

TypeId
DistributedMlTcpAgent::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::DistributedMlTcpAgent")
    .SetParent<Application> ()
    .SetGroupName("Applications")
    .AddConstructor<DistributedMlTcpAgent> ()
    .AddAttribute ("NumPackets",
                   "The number of packets the application will send",
                   UintegerValue (3),
                   MakeUintegerAccessor (&DistributedMlTcpAgent::m_nPackets),
                   MakeUintegerChecker<uint32_t> ())
    .AddAttribute ("PacketSize",
                   "The size of packets",
                   UintegerValue (516),
                   MakeUintegerAccessor (&DistributedMlTcpAgent::m_packetSize),
                   MakeUintegerChecker<uint32_t> ())
    .AddAttribute ("RemoteAddress",
                   "The destination Address of the outbound packets",
                   AddressValue (),
                   MakeAddressAccessor (&DistributedMlTcpAgent::m_remote),
                   MakeAddressChecker ())
    .AddAttribute ("RemotePort",
                   "The destination port of the outbound packets",
                   UintegerValue (0),
                   MakeUintegerAccessor (&DistributedMlTcpAgent::m_port),
                   MakeUintegerChecker<uint16_t> ())
    .AddAttribute ("DataRate",
                   "The packets trans data rate",
                   DataRateValue (DataRate ("1Mbps")),
                   MakeDataRateAccessor (&DistributedMlTcpAgent::m_dataRate),
                   MakeDataRateChecker ())
    .AddAttribute ("NumClients",
                   "The number of clients the server will connect",
                   UintegerValue (3),
                   MakeUintegerAccessor (&DistributedMlTcpAgent::m_clients),
                   MakeUintegerChecker<uint32_t> ())
    .AddAttribute ("Role",
                   "The role this agent will play",
                   StringValue ("client"),
                   MakeStringAccessor (&DistributedMlTcpAgent::m_role),
                   MakeStringChecker ())
  ;
  return tid;
}



DistributedMlTcpAgent::DistributedMlTcpAgent (uint32_t id)
{
  m_seq = 1;
  m_trigged = false;
  m_id = id;
  m_TV.Initialize();
}


DistributedMlTcpAgent::~DistributedMlTcpAgent()
{
  m_socket = 0;
}

void
DistributedMlTcpAgent::SetSocket (Ptr<Socket> socket)
{
  m_socket = socket;
}

Ptr<Socket> 
DistributedMlTcpAgent::GetSocket(void)
{
  return m_socket;

}


void
DistributedMlTcpAgent::SetAttributes (Address address, Ptr<Socket> socket, uint32_t packet_size, uint32_t num_packets, uint32_t num_clients, DataRate data_rate)
{
  m_remote = address;
  m_socket = socket;
  m_nPackets = num_packets;
  m_clients = num_clients;
  m_dataRate = data_rate;
  m_packetSize = packet_size;
  NS_LOG_INFO("Socket connect m_remote!!" << m_remote);
}


void
DistributedMlTcpAgent::SetTensor(uint64_t t_addr, uint32_t size)
{
  Buff = MlBuffer(t_addr, size);

}



/*TODO:: Currently, we're not able to use std::vector<uint64_t> as python input.
        To avoid this awkward situation, we use the std::vector<uint32_t> as input, while
        adjusting it in the python input.
*/
void
DistributedMlTcpAgent::SetModel(std::vector<uint32_t> t_addrs, std::vector<uint32_t> sizes)
{
  
  uint64_t* addrs = new uint64_t[int(t_addrs.size()/2)];
  if (!t_addrs.empty())
  {
      memcpy(addrs, &t_addrs[0], t_addrs.size()*sizeof(uint32_t));
  }

  // std::cout << "addrs: " << addrs << std::endl;

  Buff = MlBuffer(addrs, sizes);

}


void 
DistributedMlTcpAgent::SetTrigger(bool trigged){
  m_trigged = trigged;
}


bool
DistributedMlTcpAgent::GetTrigger(void){
  return m_trigged;
}

void 
DistributedMlTcpAgent::SetId(uint32_t id){
  m_id = id;
}


uint32_t
DistributedMlTcpAgent::GetId(void){
  return m_id;
}


void 
DistributedMlTcpAgent::SetRole(std::string role){
  m_role = role;
}

std::string 
DistributedMlTcpAgent::GetRole(void){
  return m_role;
}


void 
DistributedMlTcpAgent::SetEnergy(Ptr<MlDeviceEnergyModel> mlEnergy){

  m_mlEnergy = mlEnergy;
}


void 
DistributedMlTcpAgent::ChaneEnergyState(MlState state){
  if (m_mlEnergy){
    // if(m_id==1 || m_id==4){
    //   std::cout << "ID:: " << m_id << " Chane state: " << state << " time: " << Simulator::Now ().As (Time::S) << std::endl;
    // }
    m_mlEnergy->ChangeState(state);

  } 
}

bool
DistributedMlTcpAgent::IsServer(void){
  return m_role.compare("server")==0;
}

bool
DistributedMlTcpAgent::IsClient(void){
  return m_role.compare("client")==0;
}

void 
DistributedMlTcpAgent::EnableBroadcast(void){
  //Allow server to broadcast model packets to all clients whether received its packets or not
  m_allowBroadcast = true;
}

void 
DistributedMlTcpAgent::DisableBroadcast(void){
  //Allow server to broadcast model packets to all clients whether received its packets or not
  m_allowBroadcast = false;
}

void 
DistributedMlTcpAgent::Initialize(void){
  ChaneEnergyState(MlState::IDLE);
  
  //TODO: Depletion Callback
  // m_depletionCallback = MakeCallback (&DistributedMlTcpAgent::DepletionHandler, this);

  // m_mlEnergy->SetEnergyDepletionCallback (m_depletionCallback);


  if (Buff.size()==0){
    Ptr<Packet> packet = Create<Packet> (100);
    Buff = MlBuffer(packet);
  }

  if(IsClient()){
    NS_LOG_INFO("CLIENT:: Initialize one client...");

    NS_LOG_INFO("CLIENT:: Number of required packets: " << m_nPackets);

    m_socket->Bind ();
    m_socket->Connect (m_remote);

  }else if(IsServer()){

    NS_LOG_INFO("SERVER:: Initialize one server...");
    Buff.Zero();
    m_nPackets *= m_clients;
    NS_LOG_INFO("SERVER:: Number of required packets: " << m_nPackets);

    if (m_socket->Bind (m_remote) == -1)
    {
      printf ("Failed to bind socket");
    }

    m_socket->Listen ();

  }else{
    NS_LOG_ERROR("ERROR:: You MUST assert the role of one agent to be either server or client!!!");
  }
  
}


void
DistributedMlTcpAgent::StartApplication (void)
{
  Initialize();

  if(IsServer()){
    m_socket->SetAcceptCallback (
            MakeNullCallback<bool, Ptr<Socket>, const Address &> (),
            MakeCallback (&DistributedMlTcpAgent::HandleAccept, this));

  }
  
  m_socket->SetSendCallback(MakeCallback(&DistributedMlTcpAgent::HandleSend, this));
  m_socket->SetRecvCallback (MakeCallback (&DistributedMlTcpAgent::HandleRead, this));

}

void 
DistributedMlTcpAgent::StopApplication (void)
{

  if (m_sendEvent.IsRunning ())
  {
    Simulator::Cancel (m_sendEvent);
  }

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
    m_socket = 0;
  }
}

void 
DistributedMlTcpAgent::HandleAccept (Ptr<Socket> s, const Address& from)
{
  NS_LOG_FUNCTION (this << s << from);

  HandleSynchron(s, 0);
  m_socketList.push_back (s);
  NS_LOG_INFO("Server:: number of accepted socket: " << m_socketList.size());
  s->SetRecvCallback (MakeCallback (&DistributedMlTcpAgent::HandleRead, this));
  s->SetSendCallback(MakeCallback(&DistributedMlTcpAgent::HandleSend, this));
}


void DistributedMlTcpAgent::HandleSynchron(Ptr<Socket> socket, uint32_t availableBufferSize){
  Buff.CopyFromMem();
  NS_LOG_INFO("ID:: " << m_id << "  Calling HandleSynchron At time " << Simulator::Now ().As (Time::S));

  Buff.FedSend (socket, m_packetSize, m_dataRate);

}


void
DistributedMlTcpAgent::HandleRead (Ptr<Socket> socket)
{
  NS_LOG_FUNCTION (this << socket);

  Ptr<Packet> packet;
  Address from;
  Address localAddress;
  NS_LOG_INFO( "ID:: " << m_id << " calling HandleRead At time " << Simulator::Now ().As (Time::S) );
  while ((packet = socket->RecvFrom (from)))
  {

    NS_LOG_INFO( "ID:: " << m_id << " Received Packet At time " << Simulator::Now ().As (Time::S) );
    if (packet->GetSize () == 0)
    { //EOF
      printf("end of sending. \n");
      break;
    }
    socket->GetSockName (localAddress);
    PacketReceived (socket, packet, from, localAddress);

  }
  if(m_nPackets>0 && GetTrigger()){
    
    NS_LOG_INFO( "ID:: " << m_id << "  Calling SendPacket At time " << Simulator::Now ().As (Time::S) );
    HandleSend(socket, 0);

  }

}




void DistributedMlTcpAgent::HandleSend(Ptr<Socket> socket, uint32_t availableBufferSize){
  NS_LOG_INFO( "ID:: " << m_id << " calling HandleSend At time " << Simulator::Now ().As (Time::S) );
  if (GetTrigger()){
    ChaneEnergyState(MlState::IDLE);

    // if(m_id>0){
    //   std::cout << "ID: " << m_id << " At time " << Simulator::Now ().As (Time::S) << " Radio state is " << m_mlEnergy->GetCurrentState () << " energy consumed: " << m_mlEnergy->GetTotalEnergyConsumption () << std::endl;
    // }

    SetTrigger(false);

    Buff.PasteToMem();  //TODO: PasteToMem has something wrong

    double delay_process = Processing();

    // std::cout << "ID:: " << m_id << "delay value: " << delay_process << std::endl;

    Buff.CopyFromMem();
    
    Time tNext_process (Seconds (delay_process));
    Simulator::Schedule (tNext_process, &DistributedMlTcpAgent::ChaneEnergyState, this, MlState::BUSY);

    double delay_sleep = Sleeping();

    Time tNext_sleep (Seconds (delay_process+delay_sleep));
    Simulator::Schedule (tNext_sleep, &DistributedMlTcpAgent::SendPacket, this, socket);

  }

}


void 
DistributedMlTcpAgent::SendPacket(Ptr<Socket> socket){
  NS_LOG_INFO( "ID:: " << m_id << "after  sendpacket Now: " << Simulator::Now ().As (Time::S) );


  Address from, to;
  socket->GetSockName (from);
  socket->GetPeerName (to);

  if(IsClient()){
    m_sendEvent = Buff.FedSend (socket, m_packetSize, m_dataRate);
    m_nPackets-=1;

  }else if(IsServer()){
    if (m_allowBroadcast){

      memory_socketList = m_socketList;
      NS_LOG_INFO( "ID:: " << m_id << "  Calling m_allowBroadcast At time " << Simulator::Now ().As (Time::S) << "at size: " << m_socketList.size() );

    }
    while(!memory_socketList.empty ()) //these are accepted sockets, close them
    {
      Ptr<Socket> socket = memory_socketList.front ();      
      
      Address from, to;
      socket->GetSockName (from);
      socket->GetPeerName (to);
      
      m_sendEvent = Buff.FedSend (socket, m_packetSize, m_dataRate);

      m_nPackets-=1;
      memory_socketList.pop_front();
    }

    Buff.Zero();   //Set Buff value to be zero, to FedAvg new data
    m_TV.Initialize();
  }

  NS_LOG_INFO ("ID:: " << m_id << " Finished SendPacket At time: " << Simulator::Now ().As (Time::S));

}


void
DistributedMlTcpAgent::PacketReceived (Ptr<Socket> socket, const Ptr<Packet> &p, const Address &from,
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

      if(IsClient()){
        m_seq = Buff.FedUpdate(complete);
      }else if(IsServer()){
        m_seq = Buff.FedAvg(complete);
      }
      
      if (m_seq==0)
      {      
        m_seq=1;
        m_count += 1;
        NS_LOG_INFO("ID:: " << m_id << "  SetTrigger At time " << Simulator::Now ().As (Time::S));
        
        if(IsServer()){
          memory_socketList.push_back(socket);
        }

        if(TriggerLogic()){
          SetTrigger(true);
        }

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



bool DistributedMlTcpAgent::TriggerLogic(){
  return true;
}



}