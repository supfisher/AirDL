#ifndef DISTRIBUTED_ML_TCP_CLIENT_H
#define DISTRIBUTED_ML_TCP_CLIENT_H


#include "ns3/application.h"
#include "ns3/event-id.h"
#include "ns3/ptr.h"
#include "ns3/address.h"
#include "ns3/traced-callback.h"
#include "ns3/socket.h"
#include "ns3/simulator.h"
#include "ns3/data-rate.h"
#include "ns3/seq-ts-size-header.h"
#include "ns3/log.h"
#include "ns3/trace-source-accessor.h"
#include "ns3/uinteger.h"

#include "ns3/distributed-ml-utils.h"
#include <vector>

namespace ns3{

class DistributedMlTcpClient : public Application 
{
public:
  static TypeId GetTypeId (void);

  DistributedMlTcpClient ();

  DistributedMlTcpClient (uint32_t ID, uint32_t PacketSize=512, uint32_t NumPackets=3, DataRate=DataRate ("1Mbps"));

  virtual ~DistributedMlTcpClient();

  void SetSocket (Ptr<Socket> socket);

  void SetTrigger(bool m_trigged);

  bool GetTrigger(void);

  void SetAttributes (Address Address, Ptr<Socket> Socket, uint32_t PacketSize=512,  uint32_t NumPackets=3, DataRate=DataRate ("1Mbps"));

  void SetTensor(uint64_t t_addr, uint32_t size);

  void SetModel(std::vector<uint32_t> t_addrs, std::vector<uint32_t> sizes);

  void SetID(uint32_t id){m_id = id;};

  uint32_t GetID(void){return m_id;};

  MlBuffer GetBuff(void){return Buff;};


protected:
  virtual void Processing(void){
    printf("CLIENT:: Processing Function\n");
  };

  virtual void StartApplication (void);
  virtual void StopApplication (void);

private:

  // pf* m_pFunc;
  void Initialize(void);

  void HandleRead (Ptr<Socket> socket);
  void PacketReceived (const Ptr<Packet> &p, const Address &from,
                            const Address &localAddress);

  void HandleSend(Ptr<Socket> socket, uint32_t availableBufferSize);

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
  Address         m_peer;
  uint16_t        m_port;
  uint32_t        m_packetSize;
  uint32_t        m_nPackets;
  DataRate        m_dataRate;

  EventId         m_sendEvent;

  uint32_t        m_packetsSent;

  uint32_t        m_seq;   //if received header seq==0, wait for FedSend
  // whether FedSend is trigged, we consider that client may be delayed by random event.
  // Therefore, we use m_seq==0 && m_trigged to trig FedSend
  bool            m_trigged=false;  

  Ptr<Packet>     m_packet;
  int actual = 0;
  Ptr<Packet>     m_fragment;

  SeqTsSizeHeader m_header;

  MlBuffer        Buff = MlBuffer();

  uint32_t        m_id = 0;
  
};


}

#endif