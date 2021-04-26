#include "distributed-ml-network-helper.h"


using namespace ns3;

void SpectrumWifiHelper::SetStandard(string standard){
	if (standard == "11a")
    {
      wifi.SetStandard (WIFI_STANDARD_80211a);
      ssid = Ssid ("ns380211a");
      dataRate = "OfdmRate6Mbps";
      freq = 5180;
      if (bw != 20)
        {
          std::cout << "Bandwidth is not compatible with standard" << std::endl;
          return 1;
        }
    }
  else if (standard == "11p_10MHZ")
    {
      wifi.SetStandard (WIFI_STANDARD_80211p);
      ssid = Ssid ("ns380211p_10MHZ");
      dataRate = "OfdmRate3MbpsBW10MHz";
      freq = 5860;
      dataStartTime = MicroSeconds (1400);
      dataDuration = MicroSeconds (600);
      if (bw != 10)
        {
          std::cout << "Bandwidth is not compatible with standard" << std::endl;
          return 1;
        }
    }
  else if (standard == "11p_5MHZ")
    {
      wifi.SetStandard (WIFI_STANDARD_80211p);
      ssid = Ssid ("ns380211p_5MHZ");
      dataRate = "OfdmRate1_5MbpsBW5MHz";
      freq = 5860;
      dataStartTime = MicroSeconds (2500);
      dataDuration = MicroSeconds (1200);
      if (bw != 5)
        {
          std::cout << "Bandwidth is not compatible with standard" << std::endl;
          return 1;
        }
    }
  else if (standard == "11n_2_4GHZ")
    {
      wifi.SetStandard (WIFI_STANDARD_80211n_2_4GHZ);
      ssid = Ssid ("ns380211n_2_4GHZ");
      dataRate = "HtMcs0";
      freq = 2402 + (bw / 2); //so as to have 2412/2422 for 20/40
      dataStartTime = MicroSeconds (4700);
      dataDuration = MicroSeconds (400);
      if (bw != 20 && bw != 40)
        {
          std::cout << "Bandwidth is not compatible with standard" << std::endl;
          return 1;
        }
    }
  else if (standard == "11n_5GHZ")
    {
      wifi.SetStandard (WIFI_STANDARD_80211n_5GHZ);
      ssid = Ssid ("ns380211n_5GHZ");
      dataRate = "HtMcs0";
      freq = 5170 + (bw / 2); //so as to have 5180/5190 for 20/40
      dataStartTime = MicroSeconds (1000);
      if (bw != 20 && bw != 40)
        {
          std::cout << "Bandwidth is not compatible with standard" << std::endl;
          return 1;
        }
    }
  else if (standard == "11ac")
    {
      wifi.SetStandard (WIFI_STANDARD_80211ac);
      ssid = Ssid ("ns380211ac");
      dataRate = "VhtMcs0";
      freq = 5170 + (bw / 2); //so as to have 5180/5190/5210/5250 for 20/40/80/160
      dataStartTime = MicroSeconds (1100);
      dataDuration += MicroSeconds (400); //account for ADDBA procedure
      if (bw != 20 && bw != 40 && bw != 80 && bw != 160)
        {
          std::cout << "Bandwidth is not compatible with standard" << std::endl;
          return 1;
        }
    }
  else if (standard == "11ax_2_4GHZ")
    {
      wifi.SetStandard (WIFI_STANDARD_80211ax_2_4GHZ);
      ssid = Ssid ("ns380211ax_2_4GHZ");
      dataRate = "HeMcs0";
      freq = 2402 + (bw / 2); //so as to have 2412/2422/2442 for 20/40/80
      dataStartTime = MicroSeconds (5500);
      dataDuration += MicroSeconds (2000); //account for ADDBA procedure
      if (bw != 20 && bw != 40 && bw != 80)
        {
          std::cout << "Bandwidth is not compatible with standard" << std::endl;
          return 1;
        }
    }
  else if (standard == "11ax_5GHZ")
    {
      wifi.SetStandard (WIFI_STANDARD_80211ax_5GHZ);
      ssid = Ssid ("ns380211ax_5GHZ");
      dataRate = "HeMcs0";
      freq = 5170 + (bw / 2); //so as to have 5180/5190/5210/5250 for 20/40/80/160
      dataStartTime = MicroSeconds (1200);
      dataDuration += MicroSeconds (500); //account for ADDBA procedure
      if (bw != 20 && bw != 40 && bw != 80 && bw != 160)
        {
          std::cout << "Bandwidth is not compatible with standard" << std::endl;
          return 1;
        }
    }
  else
    {
      std::cout << "Unknown OFDM standard (please refer to the listed possible values)" << std::endl;
      return 1;
    }
}


Ptr<SpectrumChannel> SpectrumWifiHelper::SetChannel(){
	/* channel and propagation */
	SpectrumChannelHelper channelHelper = SpectrumChannelHelper::Default ();
	channelHelper.SetChannel ("ns3::MultiModelSpectrumChannel");
	// constant path loss added just to show capability to set different propagation loss models
	// FriisSpectrumPropagationLossModel already added by default in SpectrumChannelHelper
	channelHelper.AddSpectrumPropagationLoss ("ns3::ConstantSpectrumPropagationLossModel");
	Ptr<SpectrumChannel> channel = channelHelper.Create ();
  return channel;
}

  
void SpectrumWifiHelper::SetWifi(){
  /* Wi-Fi transmitter setup */
  SpectrumWifiPhyHelper spectrumPhy;
  spectrumPhy.SetChannel (channel);
  spectrumPhy.SetErrorRateModel ("ns3::NistErrorRateModel");
  spectrumPhy.Set ("Frequency", UintegerValue (freq));
  spectrumPhy.Set ("ChannelWidth", UintegerValue (bw));
  spectrumPhy.Set ("TxPowerStart", DoubleValue (pow)); // dBm
  spectrumPhy.Set ("TxPowerEnd", DoubleValue (pow));

  WifiMacHelper mac;
  wifi.SetRemoteStationManager ("ns3::ConstantRateWifiManager","DataMode", StringValue (dataRate),
                                "ControlMode", StringValue (dataRate));

  mac.SetType ("ns3::StaWifiMac",
               "Ssid", SsidValue (ssid),
               "ActiveProbing", BooleanValue (false));
  NetDeviceContainer staDevice = wifi.Install (spectrumPhy, mac, wifiStaNode);
  mac.SetType ("ns3::ApWifiMac",
               "Ssid", SsidValue (ssid),
               "EnableBeaconJitter", BooleanValue (false)); // so as to be sure that first beacon arrives quickly
  NetDeviceContainer apDevice = wifi.Install (spectrumPhy, mac, wifiApNode);

}
  