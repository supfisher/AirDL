#include "ns3/gnuplot.h"
#include "ns3/command-line.h"
#include "ns3/string.h"
#include "ns3/ssid.h"
#include "ns3/spectrum-helper.h"
#include "ns3/spectrum-wifi-helper.h"
#include "ns3/spectrum-analyzer-helper.h"
#include "ns3/spectrum-channel.h"
#include "ns3/mobility-helper.h"

using namespace ns3;

class SpectrumWifiHelper{
public:
	SpectrumWifiHelper(){};
	virtual ~SpectrumWifiHelper(){};

	void SetStandard(string standard);

	void SetWifi();

private:
	WifiHelper wifi;
	Ssid ssid;
	std::string dataRate;
	int freq;
	Time dataStartTime = MicroSeconds (800); // leaving enough time for beacon and association procedure
	Time dataDuration = MicroSeconds (300); // leaving enough time for data transfer (+ acknowledgment)
}