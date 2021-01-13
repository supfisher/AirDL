#ifndef DISTRIBUTED_ML_TCP_SERVER_H
#define DISTRIBUTED_ML_TCP_SERVER_H


#include "ns3/application.h"
#include "ns3/event-id.h"
#include "ns3/ptr.h"
#include "ns3/traced-callback.h"
#include "ns3/address.h"
#include "ns3/inet-socket-address.h"
#include "ns3/seq-ts-size-header.h"
#include <unordered_map>

#include "ns3/distributed-ml-utils.h"

#include "tcp-socket.h"

#include <map>
#include <algorithm>

namespace ns3{
class Address;
class Socket;
class Packet;



class TracedVariables{
public:
  TracedVariables(uint32_t& count, uint32_t& seq, bool& trigged): Count(&count), Seq(&seq), Trigged(&trigged){};

  void Initialize(void){
    *Count = 0;
    *Seq = 1;
    *Trigged = false;
  }

  int get(std::string key){
    transform(key.begin(),key.end(),key.begin(),::tolower);
    switch (KV[key]){
      case 0:
        return (int)(*Count);
      case 1:
        return (int)(*Seq);
      case 2:
        return (int)(*Trigged);
      default:
        printf("Wrongly Input Value: %s. Please Input string value selected from 'count', 'seq' and 'trigged'\n", key.c_str());
        return -1;
    }
    // if(key=='count'){
    //   return (int)(*Count);
    // }else if(key=='seq'){
    //   return (int)(*Seq);
    // }else if(key=='trigged'){
    //   return (int)(*Trigged);
    // }else{
    //   printf("Wrongly Input Value: %s. Please Input string value selected from 'count', 'seq' and 'trigged'\n", key.c_str());
    //   return -1;
    // }
  }


private:
  uint32_t* Count;
  uint32_t* Seq;
  bool*     Trigged;
  std::map<std::string, int> KV = {
    {"count",0}, {"seq",1}, {"trigged",2}
  };
};



class DistributedMlTcpServer : public Application 
{
public:

  static TypeId GetTypeId (void);


  DistributedMlTcpServer();
  virtual ~DistributedMlTcpServer();
  void SetSocket(Ptr<Socket> socket);

  void SetTrigger(bool m_trigged);

  bool GetTrigger(void);

  void SetAttributes (Address Address, Ptr<Socket> Socket, uint32_t PacketSize=512,  uint32_t NumPackets=3, uint32_t NumClients=3, DataRate=DataRate ("1Mbps"));

  void SetTensor(uint64_t t_addr, uint32_t size);

  void SetModel(std::vector<uint32_t> t_addrs, std::vector<uint32_t> sizes);

  TracedVariables m_TV = TracedVariables(m_count, m_seq, m_trigged);

protected:

  virtual void Processing(void){
    printf("SERVER:: Processing Function\n");
  };

  virtual bool TriggerLogic(void);

  virtual void StartApplication (void);
  virtual void StopApplication (void);

private:

  void Initialize(void);

  void HandleRead (Ptr<Socket> socket);
  void HandleAccept (Ptr<Socket> s, const Address& from);
  void PacketReceived (Ptr<Socket> socket, const Ptr<Packet> &p, const Address &from,
                            const Address &localAddress);

  void SendPacket();


  Ptr<Socket>     m_socket;
  std::list<Ptr<Socket> > m_socketList;
  DataRate        m_dataRate;
  uint32_t        m_nPackets;
  uint32_t        m_packetSize;
  Address         m_remote;
  uint16_t        m_port;
  uint64_t        m_totalRx=0;      //!< Total bytes received
  

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


  MlBuffer Buff;
  uint32_t        m_seq=1;
  bool            m_trigged=false;
  Ptr<Packet>     m_packet;
  SeqTsSizeHeader m_header;

  bool            m_sync=true; //whethere set a syncronous distributed ml
  uint32_t        m_clients;  //the number of clients that server has to received, if set set syncronous
  uint32_t        m_count=0;  // The number of sucssed transimission
  std::list<Ptr<Socket>>  memory_socketList; // socket list to record packet from socket

};

}
#endif