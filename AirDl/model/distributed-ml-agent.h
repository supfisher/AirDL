#ifndef DISTRIBUTED_ML_TCP_AGENT_H
#define DISTRIBUTED_ML_TCP_AGENT_H


#include "ns3/application.h"
#include "ns3/event-id.h"
#include "ns3/ptr.h"
#include "ns3/address.h"
#include "ns3/traced-callback.h"
#include "ns3/socket.h"
#include "ns3/simulator.h"
#include "ns3/data-rate.h"
#include "ns3/log.h"
#include "ns3/trace-source-accessor.h"
#include "ns3/uinteger.h"
#include "ns3/string.h"
#include "ns3/seq-ts-size-header.h"
#include "ns3/wifi-radio-energy-model.h"
#include "ns3/traced-value.h"

#include "ns3/distributed-ml-utils.h"
#include <vector>
#include <unordered_map>


namespace ns3{

class DistributedMlTcpAgent : public Application 
{
public:
  static TypeId GetTypeId (void);

  DistributedMlTcpAgent (uint32_t id=0);

  virtual ~DistributedMlTcpAgent();

  void SetSocket (Ptr<Socket> socket);

  Ptr<Socket> GetSocket(void);

  void SetAttributes (Address address, Ptr<Socket> socket, uint32_t packet_size=512,  uint32_t num_packets=3, uint32_t num_clients=3, DataRate=DataRate ("10Mbps"));

  void SetTrigger(bool m_trigged);

  bool GetTrigger(void);

  void SetId(uint32_t id);

  uint32_t GetId(void);

  void SetRole(std::string role);

  std::string GetRole(void);

  void SetTensor(uint64_t t_addr, uint32_t size);

  void SetModel(std::vector<uint32_t> t_addrs, std::vector<uint32_t> sizes);

  void SetEnergy(Ptr<MlDeviceEnergyModel> devEnergy);

  void ChaneEnergyState(MlState state);

  void EnableBroadcast(void);

  void DisableBroadcast(void);

  MlBuffer GetBuff(void){return Buff;};

  TracedVariables m_TV = TracedVariables(m_count, m_seq, m_trigged);


protected:
  virtual double Processing(void){
    std::cout << "ID:: " << m_id << "CLIENT:: Task Processing Function. Users Must Overwrite This Method." << std::endl;
    return 0.0;
  };

  virtual double Sleeping(void){
    std::cout << "ID:: " << m_id << "CLIENT:: Task Sleeping Function. Users Must Overwrite This Method." << std::endl;

    return 0.0;
  };

  virtual bool TriggerLogic(void);

  virtual void StartApplication (void);
  virtual void StopApplication (void);

private:

  void Initialize(void);

  bool IsClient(void);

  bool IsServer(void);

  void HandleAccept (Ptr<Socket> s, const Address& from);

  void HandleRead (Ptr<Socket> socket);

  void PacketReceived (Ptr<Socket> socket, const Ptr<Packet> &p, const Address &from,
                            const Address &localAddress);

  void HandleSend(Ptr<Socket> socket, uint32_t availableBufferSize);

  void SendPacket(Ptr<Socket> socket);

  void HandleSynchron(Ptr<Socket> socket, uint32_t availableBufferSize);

  struct AddressHash
  {
    size_t operator() (const Address &x) const
    {
      NS_ABORT_IF (!InetSocketAddress::IsMatchingType (x));
      InetSocketAddress a = InetSocketAddress::ConvertFrom (x);
      return std::hash<uint32_t>()(a.GetIpv4 ().Get ());
    }
  };

  std::unordered_map<Address, Ptr<Packet>, AddressHash> m_buffer; //!< Buffer for received packets


  Ptr<Socket>     m_socket;
  std::list<Ptr<Socket> > m_socketList;
  Address         m_remote;
  uint16_t        m_port;
  uint32_t        m_packetSize;
  uint32_t        m_nPackets;
  DataRate        m_dataRate;

  EventId         m_sendEvent;

  //if received header seq==0, wait for FedSend
  // whether FedSend is trigged, we consider that client may be delayed by random event.
  // Therefore, we use m_seq==0 && m_trigged to trig FedSend
  uint32_t        m_seq;
  bool            m_trigged;  
  uint32_t        m_count;
  uint32_t        m_id;
  std::string     m_role;
  uint32_t        m_clients;  //the number of clients that server has to received, if set set syncronous


  std::list<Ptr<Socket>>  memory_socketList; // socket list to record socket which has sent seq 0

  bool m_allowBroadcast = false; //if set to be true, the server will broadcast model packets to all clients

  MlBuffer        Buff = MlBuffer();

  Ptr<MlDeviceEnergyModel> m_mlEnergy;

  // WifiRadioEnergyModel::WifiRadioEnergyDepletionCallback m_depletionCallback; 
  
};




}

#endif