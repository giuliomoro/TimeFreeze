 /*
 ____  _____ _        _    
| __ )| ____| |      / \   
|  _ \|  _| | |     / _ \  
| |_) | |___| |___ / ___ \ 
|____/|_____|_____/_/   \_\

The platform for ultra-low latency audio and sensor processing

http://bela.io

A project of the Augmented Instruments Laboratory within the
Centre for Digital Music at Queen Mary University of London.
http://www.eecs.qmul.ac.uk/~andrewm

(c) 2016 Augmented Instruments Laboratory: Andrew McPherson,
  Astrid Bin, Liam Donovan, Christian Heinrichs, Robert Jack,
  Giulio Moro, Laurel Pardue, Victor Zappi. All rights reserved.

The Bela software is distributed under the GNU Lesser General Public License
(LGPL 3.0), available here: https://www.gnu.org/licenses/lgpl-3.0.txt
*/


#include <Bela.h>
#include <ne10/NE10.h>					// NEON FFT library
#include "SampleData.h"
#include <Midi.h>
#include <math_neon.h>

#define BUFFER_SIZE 16384

typedef struct _FreezeFft{
	float amplitude;
	float phase;
	float phaseInc;
} FreezeFft;

// TODO: your buffer and counter go here!
float gInputBuffer[BUFFER_SIZE];
int gInputBufferPointer = 0;
float gOutputBuffer[BUFFER_SIZE];
int gOutputBufferWritePointer = 0;
int gOutputBufferReadPointer = 0;
int gSampleCount = 0;

float *gWindowBuffer;

// -----------------------------------------------
// These variables used internally in the example:
const int gFFTSize = 4096;
int gHopSize = 1024;
int gPeriod = 1024;
FreezeFft gFreezeFft[gFFTSize];
float omega[gFFTSize] = {0};
float gFFTScaleFactor = 0;

int gElapsed = 0;
int gElapsedCopy;
bool gFftIsDone = true;


// FFT vars
ne10_fft_cpx_float32_t* timeDomainIn;
ne10_fft_cpx_float32_t* timeDomainOut;
ne10_fft_cpx_float32_t* frequencyDomain;
ne10_fft_cfg_float32_t cfg;

// Sample info
SampleData gSampleData;	// User defined structure to get complex data from main
int gReadPtr = 0;		// Position of last read sample from file

// Auxiliary task for calculating FFT
AuxiliaryTask gFFTTask;
int gFFTInputBufferPointer = 0;
int gFFTOutputBufferPointer = 0;

void process_fft_background(void*);


enum{
	kBypass,
	kRobot,
	kWhisper,
	kFreeze,
};
int gEffect = kFreeze; // change this here or with midi CC

float gDryWet = 1; // mix between the unprocessed and processed sound
float gPlaybackLive = 0; // mix between the file playback and the live audio input
float gGain = 300; // overall gain
float *gInputAudio = NULL;
Midi midi;


void midiCallback(MidiChannelMessage message, void* arg){
	if(message.getType() == kmmNoteOn){
		if(message.getDataByte(1) > 0){
			int note = message.getDataByte(0);
			float frequency = powf(2, (note-69)/12.f)*440;
			gPeriod = (int)(44100 / frequency + 0.5);
			printf("\nnote: %d, frequency: %f, hop: %d\n", note, frequency, gPeriod);
		}
	}

	bool shouldPrint = false;
	if(message.getType() == kmmControlChange){
		float data = message.getDataByte(1) / 127.0f;
		switch (message.getDataByte(0)){
		case 2 :
			gEffect = (int)(data * 2 + 0.5); // CC2 selects an effect between 0,1,2
			break;
		case 3 :
			gPlaybackLive = data;
			break;
		case 4 :
			gDryWet = data;
			break;
		case 5:
			gGain = data*10;
			break;
		default:
			shouldPrint = true;
		}
	}
	if(shouldPrint){
		message.prettyPrint();
	}
}

float princarg(float phase)
{
	// Alternative to
	// fmodf(phase + (float)M_PI, (float)(-2 * M_PI)) + M_PI;
	// This is faster only when abs(phase) is "small enough"
	while(phase >= M_PI)
		phase -= (float)2 * (float)M_PI;
	while(phase < -M_PI)
		phase += (float)2 * (float)M_PI;
	return phase;
}


unsigned int gLed1Out = 4;

// userData holds an opaque pointer to a data structure that was passed
// in from the call to initAudio().
//
// Return true on success; returning false halts the program.
bool setup(BelaContext* context, void* userData)
{
	pinMode(context, 0, 2, OUTPUT);
	pinMode(context, 0, 3, OUTPUT);
	pinMode(context, 0, gLed1Out, OUTPUT);
	pinMode(context, 0, gLed1Out + 1, OUTPUT);
	digitalWrite(context, 0, 2, 0);
	digitalWrite(context, 0, 3, 1);
    // Check that we have the same number of inputs and outputs.
	if(context->audioInChannels != context->audioOutChannels ||
			context->analogInChannels != context-> analogOutChannels){
		printf("Error: for this project, you need the same number of input and output channels.\n");
		return false;
	}
	midi.readFrom(0);
	midi.setParserCallback(midiCallback);
	// Retrieve a parameter passed in from the initAudio() call
	gSampleData = *(SampleData *)userData;

	gFFTScaleFactor = 1.0f / (float)gFFTSize;
	gOutputBufferWritePointer += gHopSize;
	timeDomainIn = (ne10_fft_cpx_float32_t*) NE10_MALLOC (gFFTSize * sizeof (ne10_fft_cpx_float32_t));
	timeDomainOut = (ne10_fft_cpx_float32_t*) NE10_MALLOC (gFFTSize * sizeof (ne10_fft_cpx_float32_t));
	frequencyDomain = (ne10_fft_cpx_float32_t*) NE10_MALLOC (gFFTSize * sizeof (ne10_fft_cpx_float32_t));
	cfg = ne10_fft_alloc_c2c_float32_neon (gFFTSize);

	memset(timeDomainOut, 0, gFFTSize * sizeof (ne10_fft_cpx_float32_t));
	memset(gOutputBuffer, 0, BUFFER_SIZE * sizeof(float));

	// Allocate buffer to mirror and modify the input
	gInputAudio = (float *)malloc(context->audioFrames * context->audioOutChannels * sizeof(float));
	if(gInputAudio == 0)
		return false;

	// Allocate the window buffer based on the FFT size
	gWindowBuffer = (float *)malloc(gFFTSize * sizeof(float));
	if(gWindowBuffer == 0)
		return false;

	for(int n = 0; n < gFFTSize; n++) {
		// Calculate a Hann window
		gWindowBuffer[n] = 0.5 * (1.0 - cos(2.0 * M_PI * n / (float)(gFFTSize - 1)));
		// compute the central frequency for each bin
		omega[n] = princarg(2.f*(float)M_PI*(float)gHopSize*(float)n/(float)gFFTSize);
	}

	// Initialise auxiliary tasks
	if((gFFTTask = Bela_createAuxiliaryTask(&process_fft_background, 90, "fft-calculation")) == 0)
		return false;
	rt_printf("You are listening to an FFT phase-vocoder with overlap-and-add.\n"
	"Use Midi Control Change to control:\n"
	"CC 2: effect type (bypass/robotization/whisperization)\n"
	"CC 3: mix between recorded sample and live audio input\n"
	"CC 4: mix between the unprocessed and processed sound\n"
	"CC 5: gain\n"
	);
	return true;
}

bool gShouldFreeze = false;
bool gFreezeReplace = true;
// temp array where to store the phases for using vectorized sin/cos computations;
float cosPh[gFFTSize];
float sinPh[gFFTSize];
float gFreezeTdIn[2][gFFTSize] = {{0}};

// This function handles the FFT processing in this example once the buffer has
// been assembled.
void process_fft(float *inBuffer, int inWritePointer, float *outBuffer, int outWritePointer)
{
	static int gFreezeStatus = -1;
	if(gShouldFreeze)
	{
		gShouldFreeze = false;
		gFreezeStatus = 0;
	}
	if(gFreezeStatus == 0 || gFreezeStatus == 1)
	{
		if(gFreezeReplace)
		{
			// store a copy of the input buffer for future adds
			memcpy(gFreezeTdIn[gFreezeStatus], inBuffer, sizeof(float) * gFFTSize);
		} else {
			int pointer = (inWritePointer - gFFTSize + BUFFER_SIZE) % BUFFER_SIZE;
			for(int n = 0; n < gFFTSize; n++) {
				inBuffer[pointer] = gFreezeTdIn[gFreezeStatus][n] + inBuffer[pointer];
				pointer++;
				if(pointer >= BUFFER_SIZE)
					pointer = 0;
			}
		}
	}
	
	if(gEffect != kFreeze || gFreezeStatus < 2)
	{
		// Copy buffer into FFT input
		int pointer = (inWritePointer - gFFTSize + BUFFER_SIZE) % BUFFER_SIZE;
		for(int n = 0; n < gFFTSize; n++) {
			timeDomainIn[n].r = (ne10_float32_t) inBuffer[pointer] * gWindowBuffer[n];
			timeDomainIn[n].i = 0;
			pointer++;
			if(pointer >= BUFFER_SIZE)
				pointer = 0;
		}
	
		// Run the FFT
		ne10_fft_c2c_1d_float32_neon (frequencyDomain, timeDomainIn, cfg, 0);
	}
	switch (gEffect){
	case kRobot :
		// Robotise the output
		for(int n = 0; n < gFFTSize; n++) {
			float amplitude = sqrtf(frequencyDomain[n].r * frequencyDomain[n].r + frequencyDomain[n].i * frequencyDomain[n].i);
			frequencyDomain[n].r = amplitude;
			frequencyDomain[n].i = 0;
		}
	break;
	case kWhisper :
		for(int n = 0; n < gFFTSize; n++) {
			float amplitude = sqrtf_neon(frequencyDomain[n].r * frequencyDomain[n].r + frequencyDomain[n].i * frequencyDomain[n].i);
			float phase = rand()/(float)RAND_MAX * 2 * M_PI;
			frequencyDomain[n].r = cosf_neon(phase) * amplitude;
			frequencyDomain[n].i = sinf_neon(phase) * amplitude;
		}
	break;
	case kBypass:
		//bypass
	break;
	case kFreeze:
		if(gFreezeStatus == 0) // initializing upon first FFT
		{
			for(int n = 0; n < gFFTSize; n++) {
				// store the amplitude
				float amplitude = sqrtf(frequencyDomain[n].r * frequencyDomain[n].r + frequencyDomain[n].i * frequencyDomain[n].i);
				gFreezeFft[n].amplitude = amplitude;
				// init the phase
				gFreezeFft[n].phase = 0;
			}
		}
		if(gFreezeStatus == 0 || gFreezeStatus == 1)
		{
			// analyze the phase for the first two grains
			for(int n = 0; n < gFFTSize; n++) {
		 		// compute phase;
				float phase = atan2f(frequencyDomain[n].i, frequencyDomain[n].r);
				// compute phase increment
				gFreezeFft[n].phaseInc = princarg(omega[n] + phase - gFreezeFft[n].phase - omega[n]);
			    gFreezeFft[n].phase = phase;
			}
		}
		if(gFreezeStatus >= 2) // compute updated phase
		{
			#undef USE_MATH_NEON_VEC
			#ifdef USE_MATH_NEON_VEC
			for(int n = 0; n < gFFTSize; n++) {
				float phase = gFreezeFft[n].phase;
				// add phase increment to phase
				phase = princarg(phase + gFreezeFft[n].phaseInc);
				
				// compute the output frame
				// We could normally do this:
				// frequencyDomain[n].r = cosf_neon(phase) * amplitude;
				// frequencyDomain[n].i = sinf_neon(phase) * amplitude;
				// ... instead we use the vectorized version,
				// so we first collect all the phases we need:
				// only sinfv is available, so turn the cos argument into the 
				// corresponding sin argument: cos(ph) = sin(ph + pi/2)
				cosPh[n] = phase + (float)M_PI * 0.5f;
				sinPh[n] = phase;
				// store the phase for the next iteration
				gFreezeFft[n].phase = phase;
			}
			// compute the sin and cos in place
			sinfv_neon(cosPh, gFFTSize, cosPh);
			sinfv_neon(sinPh, gFFTSize, sinPh);
			// store the results
			for(int n = 0; n < gFFTSize; ++n) {
				float amplitude = gFreezeFft[n].amplitude;
				frequencyDomain[n].r = cosPh[n] * amplitude;
				frequencyDomain[n].i = sinPh[n] * amplitude;
			}
		#else /* USE_MATH_NEON_VEC */
			for(int n = 0; n < gFFTSize; n++) {
				float phase = gFreezeFft[n].phase;
				float amplitude = gFreezeFft[n].amplitude;
				// add phase increment to phase
				phase = princarg(phase + gFreezeFft[n].phaseInc);
				
				frequencyDomain[n].r = cosf_neon(phase) * amplitude;
				frequencyDomain[n].i = sinf_neon(phase) * amplitude;
				gFreezeFft[n].phase = phase;
			}
		#endif
		}
		++gFreezeStatus;
	break;
	}

	// Run the inverse FFT
	ne10_fft_c2c_1d_float32_neon (timeDomainOut, frequencyDomain, cfg, 1);
	// Overlap-and-add timeDomainOut into the output buffer
	int pointer = outWritePointer;
	for(int n = 0; n < gFFTSize; n++) {
		outBuffer[pointer] += (timeDomainOut[n].r) * gFFTScaleFactor;
		if(isnan(outBuffer[pointer]))
			rt_printf("outBuffer OLA\n");
		pointer++;
		if(pointer >= BUFFER_SIZE)
			pointer = 0;
	}
	gFftIsDone = true;
}

// Function to process the FFT in a thread at lower priority
void process_fft_background(void*) {
	process_fft(gInputBuffer, gFFTInputBufferPointer, gOutputBuffer, gFFTOutputBufferPointer);
}

// render() is called regularly at the highest priority by the audio engine.
// Input and output are given from the audio hardware and the other
// ADCs and DACs (if available). If only audio is available, numMatrixFrames
// will be 0.
void render(BelaContext* context, void* userData)
{
    gDryWet = analogRead(context, 0, 0);
	gPlaybackLive = analogRead(context, 0, 1);
	static int count = 0;
	count++;
	if(count % 2000 == 0)
	{
		// rt_printf("In: %d %d", digitalRead(context, 0, 0), digitalRead(context, 0, 1));
		// rt_printf("\n");
	}	

	float* audioOut = context->audioOut;
	int numAudioFrames = context->audioFrames;
	int numAudioChannels = context->audioOutChannels;
	// ------ this code internal to the demo; leave as is ----------------

	for(unsigned int n = 0; n < context->digitalFrames; ++n)
	{
		// receive trigger input
		static int ledOn[2] = {0};
		static bool oldIn[2] = {false};
		int ledDuration = 5000;
		int freezeTriggerIn = 0;
		for(unsigned int c = 0; c < 2; ++c)
		{
			bool in = digitalRead(context, n, freezeTriggerIn + c);
			if(in && !oldIn[c])
			{
				gFreezeReplace = c == 0 ? true : false;
				gShouldFreeze = true;
				ledOn[c] = ledDuration;
				digitalWrite(context, n, gLed1Out + c, 1);
			}
			oldIn[c] = in;
			--ledOn[c];
			if(ledOn[c] == 0)
				digitalWrite(context, n, gLed1Out + c, 0);
		}
	}
	
	// Prep the "input" to be the sound file played in a loop
	for(int n = 0; n < numAudioFrames; n++) {
		static int count = -gHopSize * 3;
		++count;
		if(count % ((int)(context->audioSampleRate) * 3) == 0)
		{
			static int count = 0;
			++count;
			// rt_printf("%d Freezing\n", count);
			// gShouldFreeze = true;
		}
		if(gReadPtr < gSampleData.sampleLen)
			gInputAudio[2*n] = gInputAudio[2*n+1] = gSampleData.samples[gReadPtr]*(1-gPlaybackLive) +
				gPlaybackLive*0.5f*(audioRead(context,n,0)+audioRead(context,n,1));
		else
			gInputAudio[2*n] = gInputAudio[2*n+1] = 0;
		if(++gReadPtr >= gSampleData.sampleLen)
			gReadPtr = 0;

	}
	// -------------------------------------------------------------------

	for(int n = 0; n < numAudioFrames; n++) {
		gInputBuffer[gInputBufferPointer] = ((gInputAudio[n*numAudioChannels] + gInputAudio[n*numAudioChannels+1]) * 0.5);

		// Copy output buffer to output
		for(int channel = 0; channel < numAudioChannels; channel++){
			audioOut[n * numAudioChannels + channel] = gOutputBuffer[gOutputBufferReadPointer] * gGain * gDryWet + (1 - gDryWet) * gInputAudio[n * numAudioChannels + channel];
		}

		// Clear the output sample in the buffer so it is ready for the next overlap-add
		gOutputBuffer[gOutputBufferReadPointer] = 0;
		gOutputBufferReadPointer++;
		if(gOutputBufferReadPointer >= BUFFER_SIZE)
			gOutputBufferReadPointer = 0;
		gOutputBufferWritePointer++;
		if(gOutputBufferWritePointer >= BUFFER_SIZE)
			gOutputBufferWritePointer = 0;

		gInputBufferPointer++;
		if(gInputBufferPointer >= BUFFER_SIZE)
			gInputBufferPointer = 0;

		++gSampleCount;
		++gElapsed;
		if(gSampleCount >= gHopSize) {
			if(!gFftIsDone)
			{
				static int count = 0;
				++count;
				rt_printf("%d FFT did not complete on time.\n", count);
			}
			gFftIsDone = false;
			//process_fft(gInputBuffer, gInputBufferPointer, gOutputBuffer, gOutputBufferPointer);
			gElapsedCopy = gElapsed;
			gFFTInputBufferPointer = gInputBufferPointer;
			gFFTOutputBufferPointer = gOutputBufferWritePointer;
			Bela_scheduleAuxiliaryTask(gFFTTask);
			gSampleCount = 0;
		}
	}
	gHopSize = gPeriod;
}

// cleanup_render() is called once at the end, after the audio has stopped.
// Release any resources that were allocated in initialise_render().

void cleanup(BelaContext* context, void* userData)
{
	NE10_FREE(timeDomainIn);
	NE10_FREE(timeDomainOut);
	NE10_FREE(frequencyDomain);
	NE10_FREE(cfg);
	free(gInputAudio);
	free(gWindowBuffer);
}


/**
\example FFT-phase-vocoder/render.cpp

Phase Vocoder
----------------------

This sketch shows an implementation of a phase vocoder and builds on the previous FFT example.
Again it uses the NE10 library, included at the top of the file.

Read the documentation on the NE10 library [here](http://projectne10.github.io/Ne10/doc/annotated.html).
*/
