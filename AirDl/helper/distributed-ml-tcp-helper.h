/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2008 INRIA
 *
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
 *
 * Author: Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 */
#ifndef DISTRIBUTED_ML_TCP_HELPER_H
#define DISTRIBUTED_ML_TCP_HELPER_H

#include <stdint.h>
#include "ns3/application-container.h"
#include "ns3/node-container.h"
#include "ns3/object-factory.h"
#include "ns3/ipv4-address.h"
#include "ns3/ipv6-address.h"
#include "ns3/distributed-ml-agent.h"
#include "ns3/distributed-ml-utils.h"
#include "ns3/uinteger.h"
#include "ns3/names.h"


#include <vector>
#include "ns3/attribute.h"
#include "ns3/output-stream-wrapper.h"
#include "ns3/position-allocator.h"

namespace ns3 {

class DistributedMlTcpAgentHelper : public Object
{
public:
  /**
   * Create UdpEchoClientHelper which will make life easier for people trying
   * to set up simulations with echos. Use this variant with addresses that do
   * include a port value (e.g., InetSocketAddress and Inet6SocketAddress).
   *
   * \param addr The address of the remote udp echo server
   */
  DistributedMlTcpAgentHelper (std::string role="client");

  /**
   * Record an attribute to be set in each Application after it is is created.
   *
   * \param name the name of the attribute to set
   * \param value the value of the attribute to set
   */
  void SetAttribute (std::string name, const AttributeValue &value);

  /**
   * Create a udp echo client application on the specified node.  The Node
   * is provided as a Ptr<Node>.
   *
   * \param node The Ptr<Node> on which to create the UdpEchoClientApplication.
   *
   * \returns An ApplicationContainer that holds a Ptr<Application> to the 
   *          application created
   */
  ApplicationContainer Install (Ptr<Node> node) const;

  /**
   * Create a udp echo client application on the specified node.  The Node
   * is provided as a string name of a Node that has been previously 
   * associated using the Object Name Service.
   *
   * \param nodeName The name of the node on which to create the UdpEchoClientApplication
   *
   * \returns An ApplicationContainer that holds a Ptr<Application> to the 
   *          application created
   */
  ApplicationContainer Install (std::string nodeName) const;

  /**
   * \param c the nodes
   *
   * Create one udp echo client application on each of the input nodes
   *
   * \returns the applications created, one application per input node.
   */
  ApplicationContainer Install (NodeContainer c) const;

  Ptr<Socket> CreateSocket(Ptr<Node> node) const;

  void SetAttributes (Ptr<Application> app, Address address, Ptr<Socket> socket, uint32_t packet_size=512,  uint32_t num_packets=3, uint32_t num_clients=3, DataRate=DataRate ("10Mbps"));

  void SetAttributes (ApplicationContainer apps, Address address, Ptr<Socket> socket, uint32_t packet_size=512,  uint32_t num_packets=3, uint32_t num_clients=3, DataRate=DataRate ("10Mbps"));

  void SetTrigger (Ptr<Application> app, bool trigger);

  void SetTrigger (ApplicationContainer apps, bool trigger);

  void SetId (Ptr<Application> app, uint32_t id);

  void SetId (ApplicationContainer apps, uint32_t id);

  void SetRole (Ptr<Application> app, std::string role);

  void SetRole (ApplicationContainer apps, std::string role);

  void SetTensor (Ptr<Application> app, uint64_t t_addr, uint32_t size);

  void SetTensor (ApplicationContainer apps, uint64_t t_addr, uint32_t size);

  void SetModel (Ptr<Application> app, std::vector<uint32_t> t_addrs, std::vector<uint32_t> sizes);

  void SetModel (ApplicationContainer apps, std::vector<uint32_t> t_addrs, std::vector<uint32_t> sizes);

  virtual Ptr<DistributedMlTcpAgent> InstallPriv (Ptr<Node> node) const;

protected:

  ObjectFactory GetObjectFactory(void);

private:
  /**
   * Install an ns3::UdpEchoClient on the node configured with all the
   * attributes set with SetAttribute.
   *
   * \param node The node on which an UdpEchoClient will be installed.
   * \returns Ptr to the application installed.
   */
  
  ObjectFactory m_factory; //!< Object factory.
};


template <class T>
class AgentHelperSon: public DistributedMlTcpAgentHelper
{
  public:

  AgentHelperSon<T> (Address address): m_address(address){
    m_factory.SetTypeId (T::GetTypeId());
    // std::cout << "Calling AgentHelperSon " << std::endl;
  };

  Ptr<DistributedMlTcpAgent> InstallPriv (Ptr<Node> node) const;
  ObjectFactory m_factory = GetObjectFactory();

private:
  Address m_address;

};


template <class T>
Ptr<DistributedMlTcpAgent>
AgentHelperSon<T>::InstallPriv (Ptr<Node> node) const
{
  // printf("Calling AgentHelperSon InstallPriv...\n");
  Ptr<Socket> socket = CreateSocket (node);
  Ptr<T> app = m_factory.Create<T> ();
  app->SetAttributes (m_address, socket);
  app->SetTrigger(true);
  node->AddApplication (app);
  return app;
}



class MlDeviceEnergyModelHelper: public Object{
public:
  MlDeviceEnergyModelHelper();
  virtual ~MlDeviceEnergyModelHelper();

  void Set (std::string name, const AttributeValue &v);
  Ptr<MlDeviceEnergyModel> InstallPriv (Ptr<EnergySource> source);
  DeviceEnergyModelContainer Install (EnergySourceContainer container);
private:
  ObjectFactory m_Energy;
};

} // namespace ns3

#endif /* UDP_ECHO_HELPER_H */
