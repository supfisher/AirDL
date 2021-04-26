#ifndef DISTRIBUTED_ML_TRACES_H
#define DISTRIBUTED_ML_TRACES_H

#include "ns3/traced-callback.h"
#include "ns3/ptr.h"
#include "ns3/node.h"
#include "ns3/simulator.h"
#include "ns3/config.h"
#include "ns3/callback.h"

#include <vector>
#include "ns3/object-factory.h"
#include "ns3/attribute.h"
#include "ns3/output-stream-wrapper.h"
#include "ns3/position-allocator.h"
#include "ns3/socket.h"
#include "ns3/pcap-file-wrapper.h"
#include "ns3/energy-source.h"
#include "ns3/basic-energy-source.h"
#include "ns3/energy-source-container.h"

#include "ns3/nstime.h"
#include "ns3/wifi-radio-energy-model.h"
#include "ns3/distributed-ml-utils.h"

namespace ns3 {

class PositionAllocator;
class MobilityModel;



/**
 * \ingroup mobility
 * \brief Helper class used to assign positions and mobility models to nodes.
 *
 * MobilityHelper::Install is the most important method here.
 */
class TraceHelper
{
public:
  /**
   * Construct a Mobility Helper which is used to make life easier when working
   * with mobility models.
   */

  TraceHelper (double period=1.0){
    m_ConsumedEnergy_count = 0;
    m_RemainingEnergy_count = 0;
    m_CourseChange_count = 0;
    m_CwndChange_count = 0;
    m_RxDrop_count = 0;
    freq = period;
  };

  /**
   * Destroy a Mobility Helper
   */
  virtual ~TraceHelper (){};

  void trace_mobility(Ptr<Node> node, std::string path="mobility.txt");

  void trace_cwnd(Ptr<Socket> socket, std::string path="cwnd.txt");

  void trace_drop(Ptr<NetDevice> netdevice, std::string path="rx.pcap");

  void trace_energy_remaining(Ptr< EnergySource > source, std::string path="energy_remaining.txt");
  
  // void trace_energy_consumation(Ptr< EnergySource > source, std::string path="energy_consumation.txt");
  void trace_ml_energy_consumation(Ptr< MlDeviceEnergyModel > basicWifiRadioModelPtr, std::string path="ml_energy_consumation.txt");

  void trace_wifi_energy_consumation(Ptr< WifiRadioEnergyModel > basicWifiRadioModelPtr, std::string path="energy_consumation.txt");


  static double freq;
private:

  // void CourseChange (Ptr<OutputStreamWrapper> stream, Ptr<const MobilityModel> mobility);

//   void CwndChange (Ptr<OutputStreamWrapper> stream, uint32_t oldCwnd, uint32_t newCwnd);

//   void RxDrop (Ptr<PcapFileWrapper> file, Ptr<const Packet> p);

  // void ConsumedEnergy (Ptr<OutputStreamWrapper> stream, double oldValue, double totalEnergy);

//   void RemainingEnergy (Ptr<OutputStreamWrapper> stream, double oldValue, double remainingEnergy);
  

  // double m_ConsumedEnergy_freq;
  // double m_RemainingEnergy_freq;
  // double m_CwndChange_freq;
  // double m_CourseChange_freq;
  // double m_RxDrop_freq;

  uint64_t m_ConsumedEnergy_count;
  uint64_t m_RemainingEnergy_count;
  uint64_t m_CwndChange_count;
  uint64_t m_CourseChange_count;
  uint64_t m_RxDrop_count;
  
};


double TraceHelper::freq = 1.0;

// double TraceHelper::m_ConsumedEnergy_freq = 0;
// double TraceHelper::m_RemainingEnergy_freq = 0;
// double TraceHelper::m_CwndChange_freq = 0;
// double TraceHelper::m_CourseChange_freq = 0;
// double TraceHelper::m_RxDrop_freq = 0;

// uint64_t TraceHelper::m_ConsumedEnergy_count=0;
// uint64_t TraceHelper::m_RemainingEnergy_count=0;
// uint64_t TraceHelper::m_CwndChange_count=0;
// uint64_t TraceHelper::m_CourseChange_count=0;
// uint64_t TraceHelper::m_RxDrop_count=0;


class PyTimer
{
public:
  PyTimer(){};
  virtual ~PyTimer(){};

  static TimeWithUnit now(std::string resolution="s");

};



}


#endif