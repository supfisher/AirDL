/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */

#include "ns3/distributed-ml-utils.h"
#include <stdio.h>
#include <algorithm>
#include <numeric>

#include "ns3/simulator.h"
#include "ns3/mpi-interface.h"


namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("DistributedMlUtils");


Ptr<Packet> ToPackets(MTensor<float>& tensor){
    Ptr<Packet> packet = Create<Packet> ((uint8_t*) (tensor.data()), uint32_t(tensor.size()*sizeof(float)/sizeof(uint8_t)));
    return packet;
}

MTensor<float> PacketsTo(const Ptr<Packet>& packet){
	
    uint32_t size = packet->GetSize ();
    uint8_t* buffer = new uint8_t[size];
    packet->CopyData(buffer, size);
    return MTensor<float>((float*)buffer, uint32_t(size/sizeof(float)*sizeof(uint8_t)));
}



//Copy m_tensor from the memmory.
void MlBuffer::CopyFromMem(){
	if(m_addrs){
		uint32_t total_sizes = accumulate(m_sizes.begin(), m_sizes.end(),0);
		m_tensor = MTensor<float>(total_sizes);
		
		for (auto i=0; i<int(m_sizes.size()); i++){
			MTensor<float> mt_tmp(m_addrs[i], m_sizes[i]);
			m_tensor.append(mt_tmp);
		}
		fresh_flag = false;  // m_tensor copied value from memory, DO NOT need to fresh it from buffer

	}else{
		NS_LOG_ERROR("CopyFromMem:: Works only when m_addrs is not NULL !!! ");
	}
	
}


//Paste m_tensor to the memmory.
void MlBuffer::PasteToMem(void){
	freshMTensor();
	if(m_addrs ){
		uint32_t copy_start_point = 0;
		for (auto i=0; i<int(m_sizes.size()); i++){
			MTensor<float> mt_tmp(m_addrs[i], m_sizes[i]);
			mt_tmp.copy_from(m_tensor, copy_start_point);
			copy_start_point += m_sizes[i];
			
			// NS_LOG_INFO("Called GetTensor:: Refrsh the data with given addrs: " << mt_tmp[0] << " m_size: " << mt_tmp.size());
		}
	}else{
		NS_LOG_ERROR("PasteToMem:: Works only when m_addrs is not NULL !!! ");
	}
}


Ptr<Packet> MlBuffer::preSend(Ptr<Packet>& packet, SeqTsSizeHeader& header, uint32_t m_packetSize){
	Ptr<Packet> fragment = packet->CreateFragment (0, std::min(m_packetSize, packet->GetSize()));

	header.SetSize(fragment->GetSize()+header.GetSerializedSize ());
	
	if(packet->GetSize()<=m_packetSize){
		header.SetSeq(0);
	}

	fragment->AddHeader(header);
	return fragment;
}


void MlBuffer::afterSend(Ptr<Packet>& packet, SeqTsSizeHeader& header){
	packet->RemoveAtStart(header.GetSize()-header.GetSerializedSize ());
	
	header.SetSeq(header.GetSeq()+1);
}


void MlBuffer::BulkSend(const Ptr<Socket>& socket, Ptr<Packet>& packet, SeqTsSizeHeader& header, uint32_t m_packetSize, DataRate m_dataRate, int actual){

	Ptr<Packet> fragment = preSend(packet, header, m_packetSize);

	actual  = socket->Send (fragment);

	if(actual != -1){
		afterSend(packet, header);
	}

	
	if(packet->GetSize()>0 || actual == -1){
		Time tNext (Seconds ((fragment->GetSize() * 8) / static_cast<float> (m_dataRate.GetBitRate ())));
		Simulator::Schedule (tNext, &MlBuffer::BulkSend, this, socket, packet, header, m_packetSize, m_dataRate, actual);
	}
	
}


void MlBuffer::Zero(void){
	m_tensor.zero();
	for(auto seq=0; seq<=int(maxSeq); seq++){
		if (!seqBuffer.empty()){
			seqBuffer[seq].zero();
			seqCount[seq]=0;
		}
		
	}	
}


//TODO: The GetBuffer could be optimized... leave for future work...
MTensor<float> MlBuffer::GetBuffer(void){
	
	uint32_t size=0;
	
	for(auto seq=0; seq<=int(maxSeq); seq++){
		size += seqBuffer[seq].size(); 
	}

	float* buffer = new float[size];

	size = 0;
	
	for(auto seq=1; seq<=int(maxSeq); seq++){
		memcpy(buffer+size, seqBuffer[seq].data(), seqBuffer[seq].size()*sizeof(float));
		size += seqBuffer[seq].size();
	}
	memcpy(buffer+size, seqBuffer[0].data(), seqBuffer[0].size()*sizeof(float));
	size += seqBuffer[0].size();

	return MTensor<float>(buffer, size);
}


uint32_t MlBuffer::GetMaxSeq(void){
	
	return maxSeq;
}

bool MlBuffer::freshMTensor(void){
    if(fresh_flag || m_tensor.empty()){
        m_tensor.copy(GetBuffer());
        fresh_flag = false;
        return true;
    }else{
    	return false;
    }
}


float MlBuffer::operator[](const uint32_t i){
	freshMTensor();
    return m_tensor[i];
}


Ptr<Packet> MlBuffer::GetPacket(void){
	freshMTensor();
	return ToPackets(m_tensor);
}


uint32_t MlBuffer::size(void){
	freshMTensor();
    return m_tensor.size();
}

MTensor<float> MlBuffer::GetTensor(void){	
	freshMTensor();
	return m_tensor;
}

void MlBuffer::AvgBuff(uint32_t seq, const Ptr<Packet>& packet){

	if (seqBuffer[seq].size()==0)
	{
		NS_LOG_ERROR("AvgBuff:: Received empty Tensor at seq: " << seq);
	}

	MTensor<float> tensor = PacketsTo(packet);

	NS_ABORT_IF (seqBuffer[seq].size() != tensor.size());

	seqBuffer[seq] = (seqBuffer[seq]*seqCount[seq] + tensor)/(seqCount[seq]+1);

	seqCount[seq] += 1;
	
}


void MlBuffer::UpdateBuff(uint32_t seq, const Ptr<Packet>& packet){

	if (seqBuffer[seq].size()==0)
	{
		NS_LOG_ERROR("UpdateBuff:: Received empty Tensor at seq: " << seq);
	}

	seqBuffer[seq] = PacketsTo(packet);

}


void MlBuffer::FedSend(Ptr<Socket> socket, uint32_t m_packetSize, DataRate m_dataRate){
	SeqTsSizeHeader header;
	header.SetSeq(1);

	Ptr<Packet> packet = GetPacket();
	BulkSend(socket, packet, header, m_packetSize, m_dataRate, 0);
}


uint32_t MlBuffer::FedAvg(Ptr<Packet>& packet){
	//using map: with seq as key and packetToVec as value

	SeqTsSizeHeader header;
	packet->RemoveHeader (header);
	uint32_t seq = header.GetSeq();

	if(seq > maxSeq){
		maxSeq += 1;
	}

	if(packet->GetSize()==0){
		NS_LOG_ERROR("FedAvg::  Something wrong:  received empty packet at seq: " << seq);
	}
	
	fresh_flag=true;
	if (seqBuffer.find (seq) == seqBuffer.end ())
	{	
		seqBuffer.insert(std::make_pair (seq, PacketsTo(packet))).first;
		seqCount.insert(std::make_pair (seq, 1)).first;
	}else{
		AvgBuff(seq, packet);
	}


	return seq;
}


uint32_t MlBuffer::FedUpdate(Ptr<Packet>& packet){
	SeqTsSizeHeader header;
	packet->RemoveHeader (header);
	uint32_t seq = header.GetSeq();

	if(seq > maxSeq){
		maxSeq += 1;
	}
	
	fresh_flag=true;
	if (seqBuffer.find (seq) == seqBuffer.end ())
	{	
		seqBuffer.insert(std::make_pair (seq, PacketsTo(packet))).first;
	}else{
		UpdateBuff(seq, packet);
	}

	if(seqBuffer[seq].size() == 0){
		NS_LOG_ERROR("FedUpdate:: Received empty Tensor at seq: " << seq);
	}

	return seq;
}




}






















