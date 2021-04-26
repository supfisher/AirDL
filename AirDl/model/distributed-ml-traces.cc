#include "distributed-ml-traces.h"
#include "ns3/mobility-helper.h"
#include "ns3/mobility-model.h"
#include "ns3/position-allocator.h"
#include "ns3/hierarchical-mobility-model.h"
#include "ns3/log.h"
#include "ns3/pointer.h"
#include "ns3/config.h"
#include "ns3/simulator.h"
#include "ns3/names.h"
#include "ns3/string.h"
#include <iostream>

#include "ns3/trace-helper.h"
#include "ns3/net-device.h"


using namespace ns3;

bool
SavedFreq(double time_now, uint64_t& count){
  if (time_now>=TraceHelper::freq*count){
    count+=1;
    return true;
  }else{
    return false;
  }
}



static double
DoRound (double v)
{
  if (v <= 1e-4 && v >= -1e-4)
    {
      return 0.0;
    }
  else if (v <= 1e-3 && v >= 0)
    {
      return 1e-3;
    }
  else if (v >= -1e-3 && v <= 0)
    {
      return -1e-3;
    }
  else
    {
      return v;
    }
}


void
CourseChange (Ptr<OutputStreamWrapper> stream, uint64_t& count, Ptr<const MobilityModel> mobility)
{
  double now = Simulator::Now ().GetSeconds ();

  if(SavedFreq(now, count)){
    std::ostream* os = stream->GetStream ();
    Ptr<Node> node = mobility->GetObject<Node> ();
    *os << "now=" << Simulator::Now ().GetSeconds ()
        << " node=" << node->GetId ();
    Vector pos = mobility->GetPosition ();
    pos.x = DoRound (pos.x);
    pos.y = DoRound (pos.y);
    pos.z = DoRound (pos.z);
    Vector vel = mobility->GetVelocity ();
    vel.x = DoRound (vel.x);
    vel.y = DoRound (vel.y);
    vel.z = DoRound (vel.z);
    std::streamsize saved_precision = os->precision ();
    std::ios::fmtflags saved_flags = os->flags ();
    os->precision (3);
    os->setf (std::ios::fixed,std::ios::floatfield);
    *os << " pos=" << pos.x << ":" << pos.y << ":" << pos.z
        << " vel=" << vel.x << ":" << vel.y << ":" << vel.z
        << std::endl;
    os->flags (saved_flags);
    os->precision (saved_precision);

  }
  
}

void 
TraceHelper::trace_mobility(Ptr<Node> node, std::string path){
	// printf("Calling Trace mobility ...\n");

	AsciiTraceHelper asciiTraceHelper;
	Ptr<OutputStreamWrapper> stream = asciiTraceHelper.CreateFileStream (path);

	std::ostringstream oss;
	uint32_t nodeid = node->GetId ();
	oss << "/NodeList/" << nodeid << "/$ns3::MobilityModel/CourseChange";

	Config::ConnectWithoutContext (oss.str (), MakeBoundCallback (&CourseChange, stream, m_CourseChange_count));
}



void
CwndChange (Ptr<OutputStreamWrapper> stream, uint64_t count, uint32_t oldCwnd, uint32_t newCwnd)
{
  double now = Simulator::Now ().GetSeconds ();
  if(SavedFreq(now, count)){
    std::ostream* os = stream->GetStream ();
    std::streamsize saved_precision = os->precision ();
    std::ios::fmtflags saved_flags = os->flags ();
    os->precision (3);
    os->setf (std::ios::fixed,std::ios::floatfield);
    *os << Simulator::Now ().GetSeconds () << "\t"  << "oldCwnd=" << oldCwnd << "\t" << "newCwnd=" << newCwnd << std::endl;
    os->flags (saved_flags);
    os->precision (saved_precision);
  }

}


void 
TraceHelper::trace_cwnd(Ptr<Socket> socket, std::string path){
	// printf("Calling Trace cwnd ...\n");

  AsciiTraceHelper asciiTraceHelper;
	Ptr<OutputStreamWrapper> stream = asciiTraceHelper.CreateFileStream (path);
	socket->TraceConnectWithoutContext ("CongestionWindow", MakeBoundCallback (&CwndChange, stream, m_CwndChange_count));;
}




void
RxDrop (Ptr<PcapFileWrapper> file, uint64_t& count,  Ptr<const Packet> p)
{
  double now = Simulator::Now ().GetSeconds ();
  if(SavedFreq(now, count)){
    file->Write (Simulator::Now (), p);
  }
  
}


void 
TraceHelper::trace_drop(Ptr<NetDevice> netdevice, std::string path)
{
	// printf("Calling Trace drop ...\n");

	PcapHelper pcapHelper;
	Ptr<PcapFileWrapper> file = pcapHelper.CreateFile (path, std::ios::out, PcapHelper::DLT_PPP);
	netdevice->TraceConnectWithoutContext ("PhyRxDrop", MakeBoundCallback (&RxDrop, file, m_RxDrop_count));

}



/// Trace function for remaining energy at node.
void
RemainingEnergy (Ptr<OutputStreamWrapper> stream, uint64_t& count, double oldValue, double remainingEnergy)
{
  
  double now = Simulator::Now ().GetSeconds ();
  if(SavedFreq(now, count)){
    std::ostream* os = stream->GetStream ();
    std::streamsize saved_precision = os->precision ();
    std::ios::fmtflags saved_flags = os->flags ();
    os->precision (3);
    os->setf (std::ios::fixed,std::ios::floatfield);
    *os << Simulator::Now ().GetSeconds () << "\t"  << "s Current remaining energy = " << remainingEnergy << "J" << std::endl;
    os->flags (saved_flags);
    os->precision (saved_precision);
  }
  
}



void
TraceHelper::trace_energy_remaining(Ptr< EnergySource > source, std::string path){

  // printf("Calling trace_energy_remaining ...\n");

  AsciiTraceHelper asciiTraceHelper;
  Ptr<OutputStreamWrapper> stream = asciiTraceHelper.CreateFileStream (path);

  Ptr<BasicEnergySource> basicSourcePtr = DynamicCast<BasicEnergySource> (source);
  basicSourcePtr->TraceConnectWithoutContext ("RemainingEnergy", MakeBoundCallback (&RemainingEnergy, stream, m_RemainingEnergy_count));

}


void
ConsumedEnergy (Ptr<OutputStreamWrapper> stream, uint64_t& count, double oldValue, double totalEnergy)
{

  double now = Simulator::Now ().GetSeconds ();
  if(SavedFreq(now, count)){
    std::ostream* os = stream->GetStream ();
    std::streamsize saved_precision = os->precision ();
    std::ios::fmtflags saved_flags = os->flags ();
    os->precision (3);
    os->setf (std::ios::fixed,std::ios::floatfield);
    *os << Simulator::Now ().GetSeconds () << "\t"  << "s Total energy consumed by radio = " << totalEnergy << "J" << std::endl;
    os->flags (saved_flags);
    os->precision (saved_precision);
  }
}



// void
// TraceHelper::trace_energy_consumation(Ptr< EnergySource > source, std::string path)
// {

//   printf("Calling trace_energy_consumation ...\n");
//   AsciiTraceHelper asciiTraceHelper;
//   Ptr<OutputStreamWrapper> stream = asciiTraceHelper.CreateFileStream (path);

//   Ptr<BasicEnergySource> basicSourcePtr = DynamicCast<BasicEnergySource> (source);
//   // device energy model
//   Ptr<DeviceEnergyModel> basicRadioModelPtr =
//     basicSourcePtr->FindDeviceEnergyModels ("ns3::WifiRadioEnergyModel").Get (0);
//   NS_ASSERT (basicRadioModelPtr != NULL);

//   basicRadioModelPtr->TraceConnectWithoutContext ("TotalEnergyConsumption", MakeBoundCallback (&ConsumedEnergy, stream, m_ConsumedEnergy_count));

// }

void
TraceHelper::trace_wifi_energy_consumation(Ptr< WifiRadioEnergyModel > basicWifiRadioModelPtr, std::string path)
{

  // printf("Calling trace_energy_consumation ...\n");
  AsciiTraceHelper asciiTraceHelper;
  Ptr<OutputStreamWrapper> stream = asciiTraceHelper.CreateFileStream (path);
  Ptr< DeviceEnergyModel > basicRadioModelPtr = static_cast<Ptr< DeviceEnergyModel >> (basicWifiRadioModelPtr);

  basicRadioModelPtr->TraceConnectWithoutContext ("TotalEnergyConsumption", MakeBoundCallback (&ConsumedEnergy, stream, m_ConsumedEnergy_count));

}


void
TraceHelper::trace_ml_energy_consumation(Ptr< MlDeviceEnergyModel > basicWifiRadioModelPtr, std::string path)
{

  // printf("Calling trace_energy_consumation ...\n");
  AsciiTraceHelper asciiTraceHelper;
  Ptr<OutputStreamWrapper> stream = asciiTraceHelper.CreateFileStream (path);
  // Ptr< SimpleDeviceEnergyModel > basicRadioModelPtr = static_cast<Ptr< SimpleDeviceEnergyModel >> (basicWifiRadioModelPtr);

  basicWifiRadioModelPtr->TraceConnectWithoutContext ("TotalEnergyConsumption", MakeBoundCallback (&ConsumedEnergy, stream, m_ConsumedEnergy_count));

}



TimeWithUnit  
PyTimer::now(std::string resolution){

	if (resolution=="s"){
		return Simulator::Now ().As (Time::S);
	}
	else if(resolution=="ms"){
		return Simulator::Now ().As (Time::MS);
	}
	else if(resolution=="min"){
		return Simulator::Now ().As (Time::MIN);
	}
	else if(resolution=="us"){
		return Simulator::Now ().As (Time::US);
	}
	else if(resolution=="h"){
		return Simulator::Now ().As (Time::H);
	}
	else{
		return Simulator::Now ().As (Time::S);
	}
  
}	















