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
#include "ns3/distributed-ml-tcp-helper.h"
#include "ns3/distributed-ml-tcp-server.h"
#include "ns3/distributed-ml-tcp-client.h"
#include "ns3/distributed-ml-utils.h"
#include "ns3/uinteger.h"
#include "ns3/names.h"
#include "ns3/tcp-socket-factory.h"
#include <vector>

namespace ns3 {

DistributedMlTcpServerHelper::DistributedMlTcpServerHelper (Address address, uint16_t port)
{
  printf("Calling DistributedMlTcpServerHelper (Address address, uint16_t port)\n");
  m_factory.SetTypeId (DistributedMlTcpServer::GetTypeId ());
  SetAttribute ("RemoteAddress", AddressValue (address));
  SetAttribute ("RemotePort", UintegerValue (port));
}

DistributedMlTcpServerHelper::DistributedMlTcpServerHelper (Address address)
{
  printf("Calling DistributedMlTcpServerHelper (Address address)\n");
  m_factory.SetTypeId (DistributedMlTcpServer::GetTypeId ());
  SetAttribute ("RemoteAddress", AddressValue(address));
}


void 
DistributedMlTcpServerHelper::SetAttribute (
  std::string name, 
  const AttributeValue &value)
{
  m_factory.Set (name, value);
}

Ptr<Socket> 
DistributedMlTcpServerHelper::CreateSocket(Ptr<Node> node) const
{
  Ptr<Socket> socket = Socket::CreateSocket (node, TcpSocketFactory::GetTypeId ());
  return socket;
}



void
DistributedMlTcpServerHelper::SetTensor (Ptr<Application> app, uint64_t t_addr, uint32_t size)
{
  app->GetObject<DistributedMlTcpServer>()->SetTensor (t_addr, size);
}


void
DistributedMlTcpServerHelper::SetTensor (ApplicationContainer apps, uint64_t t_addr, uint32_t size)
{
  for (ApplicationContainer::Iterator i = apps.Begin (); i != apps.End (); ++i){
    (*i)->GetObject<DistributedMlTcpServer>()->SetTensor (t_addr, size);
  }
}

void
DistributedMlTcpServerHelper::SetModel (Ptr<Application> app, std::vector<uint32_t> t_addrs, std::vector<uint32_t> sizes)
{
  app->GetObject<DistributedMlTcpServer>()->SetModel (t_addrs, sizes);
}


void
DistributedMlTcpServerHelper::SetModel (ApplicationContainer apps, std::vector<uint32_t> t_addrs, std::vector<uint32_t> sizes)
{
  for (ApplicationContainer::Iterator i = apps.Begin (); i != apps.End (); ++i){
    (*i)->GetObject<DistributedMlTcpServer>()->SetModel (t_addrs, sizes);
  }
}


ApplicationContainer
DistributedMlTcpServerHelper::Install (Ptr<Node> node) const
{
  return ApplicationContainer (InstallPriv (node));
}


ApplicationContainer
DistributedMlTcpServerHelper::Install (std::string nodeName) const
{
  Ptr<Node> node = Names::Find<Node> (nodeName);
  return ApplicationContainer (InstallPriv (node));
}


ApplicationContainer
DistributedMlTcpServerHelper::Install (NodeContainer c) const
{
  ApplicationContainer apps;
  for (NodeContainer::Iterator i = c.Begin (); i != c.End (); ++i)
    {
      apps.Add (InstallPriv (*i));
    }

  return apps;
}


Ptr<DistributedMlTcpServer>
DistributedMlTcpServerHelper::InstallPriv (Ptr<Node> node) const
{
  Ptr<Socket> serverSocket = CreateSocket (node);

  Ptr<DistributedMlTcpServer> app = m_factory.Create<DistributedMlTcpServer> ();
  app->SetSocket(serverSocket);
  node->AddApplication (app);
  return app;
}


DistributedMlTcpClientHelper::DistributedMlTcpClientHelper (Address address, uint16_t port)
{
  m_factory.SetTypeId (DistributedMlTcpClient::GetTypeId ());
  SetAttribute ("RemoteAddress", AddressValue (address));
  SetAttribute ("RemotePort", UintegerValue (port));
}


DistributedMlTcpClientHelper::DistributedMlTcpClientHelper (Address address)
{
  m_factory.SetTypeId (DistributedMlTcpClient::GetTypeId ());
  SetAttribute ("RemoteAddress", AddressValue (address));
}

void 
DistributedMlTcpClientHelper::SetAttribute (
  std::string name, 
  const AttributeValue &value)
{
  m_factory.Set (name, value);
}

ObjectFactory
DistributedMlTcpClientHelper::GetObjectFactory(void){
  return m_factory;
}

ApplicationContainer
DistributedMlTcpClientHelper::Install (Ptr<Node> node) const
{
  return ApplicationContainer (InstallPriv (node));
}


ApplicationContainer
DistributedMlTcpClientHelper::Install (std::string nodeName) const
{
  Ptr<Node> node = Names::Find<Node> (nodeName);
  return ApplicationContainer (InstallPriv (node));
}


ApplicationContainer
DistributedMlTcpClientHelper::Install (NodeContainer c) const
{
  ApplicationContainer apps;
  for (NodeContainer::Iterator i = c.Begin (); i != c.End (); ++i)
    {
      apps.Add (InstallPriv (*i));
    }

  return apps;
}


Ptr<DistributedMlTcpClient>
DistributedMlTcpClientHelper::InstallPriv (Ptr<Node> node) const
{
  Ptr<Socket> socket = CreateSocket (node);
  Ptr<DistributedMlTcpClient> app = m_factory.Create<DistributedMlTcpClient> ();
  app->SetSocket (socket);
  node->AddApplication (app);

  return app;
}


Ptr<Socket> 
DistributedMlTcpClientHelper::CreateSocket(Ptr<Node> node) const
{
  Ptr<Socket> socket = Socket::CreateSocket (node, TcpSocketFactory::GetTypeId ());
  return socket;
}


void
DistributedMlTcpClientHelper::SetTrigger (Ptr<Application> app, bool trigger)
{
  app->GetObject<DistributedMlTcpClient>()->SetTrigger (trigger);
}


void
DistributedMlTcpClientHelper::SetTrigger (ApplicationContainer apps, bool trigger)
{
  for (ApplicationContainer::Iterator i = apps.Begin (); i != apps.End (); ++i){
    (*i)->GetObject<DistributedMlTcpClient>()->SetTrigger (trigger);
  }
}


void
DistributedMlTcpClientHelper::SetTensor (Ptr<Application> app, uint64_t t_addr, uint32_t size)
{
  app->GetObject<DistributedMlTcpClient>()->SetTensor (t_addr, size);
}


void
DistributedMlTcpClientHelper::SetTensor (ApplicationContainer apps, uint64_t t_addr, uint32_t size)
{
  for (ApplicationContainer::Iterator i = apps.Begin (); i != apps.End (); ++i){
    (*i)->GetObject<DistributedMlTcpClient>()->SetTensor (t_addr, size);
  }
}

void
DistributedMlTcpClientHelper::SetModel (Ptr<Application> app, std::vector<uint32_t> t_addrs, std::vector<uint32_t> sizes)
{
  app->GetObject<DistributedMlTcpClient>()->SetModel (t_addrs, sizes);
}


void
DistributedMlTcpClientHelper::SetModel (ApplicationContainer apps, std::vector<uint32_t> t_addrs, std::vector<uint32_t> sizes)
{
  for (ApplicationContainer::Iterator i = apps.Begin (); i != apps.End (); ++i){
    (*i)->GetObject<DistributedMlTcpClient>()->SetModel (t_addrs, sizes);
  }
}




} // namespace ns3
