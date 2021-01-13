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
#include "ns3/distributed-ml-tcp-server.h"
#include "ns3/distributed-ml-tcp-client.h"
#include "ns3/distributed-ml-utils.h"
#include "ns3/uinteger.h"
#include "ns3/names.h"

namespace ns3 {

/**
 * \ingroup udpecho
 * \brief Create a server application which waits for input UDP packets
 *        and sends them back to the original sender.
 */
class DistributedMlTcpServerHelper : public Object
{
public:
  /**
   * Create UdpEchoServerHelper which will make life easier for people trying
   * to set up simulations with echos.
   *
   * \param port The port the server will wait on for incoming packets
   */
  DistributedMlTcpServerHelper (Address ip, uint16_t port);

  DistributedMlTcpServerHelper (Address addr);

  /**
   * Record an attribute to be set in each Application after it is is created.
   *
   * \param name the name of the attribute to set
   * \param value the value of the attribute to set
   */
  void SetAttribute (std::string name, const AttributeValue &value);


  Ptr<Socket> CreateSocket(Ptr<Node> node) const;


  void SetTensor (Ptr<Application> app, uint64_t t_addr, uint32_t size);

  void SetTensor (ApplicationContainer apps, uint64_t t_addr, uint32_t size);

  void SetModel (Ptr<Application> app, std::vector<uint32_t> t_addrs, std::vector<uint32_t> sizes);

  void SetModel (ApplicationContainer apps, std::vector<uint32_t> t_addrs, std::vector<uint32_t> sizes);


  /**
   * Create a UdpEchoServerApplication on the specified Node.
   *
   * \param node The node on which to create the Application.  The node is
   *             specified by a Ptr<Node>.
   *
   * \returns An ApplicationContainer holding the Application created,
   */
  ApplicationContainer Install (Ptr<Node> node) const;

  /**
   * Create a UdpEchoServerApplication on specified node
   *
   * \param nodeName The node on which to create the application.  The node
   *                 is specified by a node name previously registered with
   *                 the Object Name Service.
   *
   * \returns An ApplicationContainer holding the Application created.
   */
  ApplicationContainer Install (std::string nodeName) const;

  /**
   * \param c The nodes on which to create the Applications.  The nodes
   *          are specified by a NodeContainer.
   *
   * Create one udp echo server application on each of the Nodes in the
   * NodeContainer.
   *
   * \returns The applications created, one Application per Node in the 
   *          NodeContainer.
   */
  ApplicationContainer Install (NodeContainer c) const;

  /**
   * Install an ns3::UdpEchoServer on the node configured with all the
   * attributes set with SetAttribute.
   *
   * \param node The node on which an UdpEchoServer will be installed.
   * \returns Ptr to the application installed.
   */
  virtual Ptr<DistributedMlTcpServer> InstallPriv (Ptr<Node> node) const;

private:
  ObjectFactory m_factory; //!< Object factory.
};

/**
 * \ingroup udpecho
 * \brief Create an application which sends a UDP packet and waits for an echo of this packet
 */
class DistributedMlTcpClientHelper : public Object
{
public:
  /**
   * Create UdpEchoClientHelper which will make life easier for people trying
   * to set up simulations with echos. Use this variant with addresses that do
   * not include a port value (e.g., Ipv4Address and Ipv6Address).
   *
   * \param ip The IP address of the remote udp echo server
   * \param port The port number of the remote udp echo server
   */
  DistributedMlTcpClientHelper (Address ip, uint16_t port);
  /**
   * Create UdpEchoClientHelper which will make life easier for people trying
   * to set up simulations with echos. Use this variant with addresses that do
   * include a port value (e.g., InetSocketAddress and Inet6SocketAddress).
   *
   * \param addr The address of the remote udp echo server
   */
  DistributedMlTcpClientHelper (Address addr);

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

  void SetTrigger (Ptr<Application> app, bool trigger);

  void SetTrigger (ApplicationContainer apps, bool trigger);

  void SetTensor (Ptr<Application> app, uint64_t t_addr, uint32_t size);

  void SetTensor (ApplicationContainer apps, uint64_t t_addr, uint32_t size);

  void SetModel (Ptr<Application> app, std::vector<uint32_t> t_addrs, std::vector<uint32_t> sizes);

  void SetModel (ApplicationContainer apps, std::vector<uint32_t> t_addrs, std::vector<uint32_t> sizes);

  virtual Ptr<DistributedMlTcpClient> InstallPriv (Ptr<Node> node) const;

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



} // namespace ns3

#endif /* UDP_ECHO_HELPER_H */
