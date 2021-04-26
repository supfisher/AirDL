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
#include "ns3/distributed-ml-utils.h"
#include "ns3/uinteger.h"
#include "ns3/names.h"
#include "ns3/tcp-socket-factory.h"
#include <vector>

#include "ns3/mobility-model.h"


namespace ns3 {

DistributedMlTcpAgentHelper::DistributedMlTcpAgentHelper (std::string role)
{
  m_factory.SetTypeId (DistributedMlTcpAgent::GetTypeId ());
  SetAttribute ("Role", StringValue (role));
}


void 
DistributedMlTcpAgentHelper::SetAttribute (
  std::string name, 
  const AttributeValue &value)
{
  m_factory.Set (name, value);
}

ObjectFactory
DistributedMlTcpAgentHelper::GetObjectFactory(void){
  return m_factory;
}

ApplicationContainer
DistributedMlTcpAgentHelper::Install (Ptr<Node> node) const
{
  return ApplicationContainer (InstallPriv (node));
}


ApplicationContainer
DistributedMlTcpAgentHelper::Install (std::string nodeName) const
{
  Ptr<Node> node = Names::Find<Node> (nodeName);
  return ApplicationContainer (InstallPriv (node));
}


ApplicationContainer
DistributedMlTcpAgentHelper::Install (NodeContainer c) const
{
  ApplicationContainer apps;
  for (NodeContainer::Iterator i = c.Begin (); i != c.End (); ++i)
    {
      apps.Add (InstallPriv (*i));
    }

  return apps;
}


Ptr<DistributedMlTcpAgent>
DistributedMlTcpAgentHelper::InstallPriv (Ptr<Node> node) const
{
  Ptr<Socket> socket = CreateSocket (node);
  Ptr<DistributedMlTcpAgent> app = m_factory.Create<DistributedMlTcpAgent> ();
  app->SetSocket (socket);
  node->AddApplication (app);

  return app;
}


Ptr<Socket> 
DistributedMlTcpAgentHelper::CreateSocket(Ptr<Node> node) const
{
  Ptr<Socket> socket = Socket::CreateSocket (node, TcpSocketFactory::GetTypeId ());
  return socket;
}


void
DistributedMlTcpAgentHelper::SetTrigger (Ptr<Application> app, bool trigger)
{
  app->GetObject<DistributedMlTcpAgent>()->SetTrigger (trigger);
}


void
DistributedMlTcpAgentHelper::SetTrigger (ApplicationContainer apps, bool trigger)
{
  for (ApplicationContainer::Iterator i = apps.Begin (); i != apps.End (); ++i){
    (*i)->GetObject<DistributedMlTcpAgent>()->SetTrigger (trigger);
  }
}


void
DistributedMlTcpAgentHelper::SetId (Ptr<Application> app, uint32_t id)
{
  app->GetObject<DistributedMlTcpAgent>()->SetId (id);
}


void
DistributedMlTcpAgentHelper::SetId (ApplicationContainer apps, uint32_t id)
{
  for (ApplicationContainer::Iterator i = apps.Begin (); i != apps.End (); ++i){
    (*i)->GetObject<DistributedMlTcpAgent>()->SetId (id);
  }
}

void
DistributedMlTcpAgentHelper::SetRole (Ptr<Application> app, std::string role)
{
  app->GetObject<DistributedMlTcpAgent>()->SetRole (role);
}


void
DistributedMlTcpAgentHelper::SetRole (ApplicationContainer apps, std::string role)
{
  for (ApplicationContainer::Iterator i = apps.Begin (); i != apps.End (); ++i){
    (*i)->GetObject<DistributedMlTcpAgent>()->SetRole (role);
  }
}

void
DistributedMlTcpAgentHelper::SetAttributes (Ptr<Application> app, Address address, Ptr<Socket> socket, uint32_t packet_size, uint32_t num_packets, uint32_t num_clients, DataRate data_rate)
{
  app->GetObject<DistributedMlTcpAgent>()->SetAttributes (address, socket, packet_size, num_packets, num_clients, data_rate);
}


void
DistributedMlTcpAgentHelper::SetAttributes (ApplicationContainer apps, Address address, Ptr<Socket> socket, uint32_t packet_size, uint32_t num_packets, uint32_t num_clients, DataRate data_rate)
{

  for (ApplicationContainer::Iterator i = apps.Begin (); i != apps.End (); ++i){
    (*i)->GetObject<DistributedMlTcpAgent>()->SetAttributes (address, socket, packet_size, num_packets, num_clients, data_rate);
  }
}


void
DistributedMlTcpAgentHelper::SetTensor (Ptr<Application> app, uint64_t t_addr, uint32_t size)
{
  app->GetObject<DistributedMlTcpAgent>()->SetTensor (t_addr, size);
}


void
DistributedMlTcpAgentHelper::SetTensor (ApplicationContainer apps, uint64_t t_addr, uint32_t size)
{
  for (ApplicationContainer::Iterator i = apps.Begin (); i != apps.End (); ++i){
    (*i)->GetObject<DistributedMlTcpAgent>()->SetTensor (t_addr, size);
  }
}

void
DistributedMlTcpAgentHelper::SetModel (Ptr<Application> app, std::vector<uint32_t> t_addrs, std::vector<uint32_t> sizes)
{
  app->GetObject<DistributedMlTcpAgent>()->SetModel (t_addrs, sizes);
}


void
DistributedMlTcpAgentHelper::SetModel (ApplicationContainer apps, std::vector<uint32_t> t_addrs, std::vector<uint32_t> sizes)
{
  for (ApplicationContainer::Iterator i = apps.Begin (); i != apps.End (); ++i){
    (*i)->GetObject<DistributedMlTcpAgent>()->SetModel (t_addrs, sizes);
  }
}



MlDeviceEnergyModelHelper::MlDeviceEnergyModelHelper ()
{
  m_Energy.SetTypeId ("ns3::MlDeviceEnergyModel");

}

MlDeviceEnergyModelHelper::~MlDeviceEnergyModelHelper ()
{
}

void
MlDeviceEnergyModelHelper::Set (std::string name, const AttributeValue &v)
{
  m_Energy.Set (name, v);
}

DeviceEnergyModelContainer
MlDeviceEnergyModelHelper::Install (EnergySourceContainer container)
{
  DeviceEnergyModelContainer modelContainer;
  for (EnergySourceContainer::Iterator i = container.Begin (); i != container.End (); ++i){
    Ptr<MlDeviceEnergyModel> model = InstallPriv((*i)->GetObject<EnergySource>());
    modelContainer.Add(model);
  }
  return modelContainer;
}

Ptr<MlDeviceEnergyModel>
MlDeviceEnergyModelHelper::InstallPriv (Ptr<EnergySource> source)
{
  Ptr<MlDeviceEnergyModel> model = m_Energy.Create ()->GetObject<MlDeviceEnergyModel> ();
  model->SetEnergySource(source);
  return model;
}


} // namespace ns3
