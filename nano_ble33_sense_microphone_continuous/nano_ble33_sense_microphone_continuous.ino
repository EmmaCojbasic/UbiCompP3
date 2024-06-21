/* Edge Impulse ingestion SDK
 * Copyright (c) 2022 EdgeImpulse Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

// If your target is limited in memory remove this macro to save 10K RAM
#define EIDSP_QUANTIZE_FILTERBANK   0

/**
 * Define the number of slices per model window. E.g. a model window of 1000 ms
 * with slices per model window set to 4. Results in a slice size of 250 ms.
 * For more info: https://docs.edgeimpulse.com/docs/continuous-audio-sampling
 */
#define EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW 4

/*
 ** NOTE: If you run into TFLite arena allocation issue.
 **
 ** This may be due to may dynamic memory fragmentation.
 ** Try defining "-DEI_CLASSIFIER_ALLOCATION_STATIC" in boards.local.txt (create
 ** if it doesn't exist) and copy this file to
 ** `<ARDUINO_CORE_INSTALL_PATH>/arduino/hardware/<mbed_core>/<core_version>/`.
 **
 ** See
 ** (https://support.arduino.cc/hc/en-us/articles/360012076960-Where-are-the-installed-cores-located-)
 ** to find where Arduino installs cores on your machine.
 **
 ** If the problem persists then there's not enough memory for this model and application.
 */

/* Includes ---------------------------------------------------------------- */
#include <Arduino_LSM9DS1.h> // Biblioteka za akcelerometar
#include <Arduino_HTS221.h>  // Biblioteka za senzor temperature i vlažnosti
#include <Arduino_LPS22HB.h> // Biblioteka za barometar
#include <PDM.h>
#include <ecojbasic-project-1_inferencing.h>

// Definicija pinova za LED diode
#define LED_PIN_RED 22
#define LED_PIN_GREEN 24
#define LED_PIN_BLUE 23
#define LED_PIN 13

/** Audio buffers, pointers and selectors */
typedef struct {
    signed short *buffers[2];
    unsigned char buf_select;
    unsigned char buf_ready;
    unsigned int buf_count;
    unsigned int n_samples;
} inference_t;

static inference_t inference;
static bool record_ready = false;
static signed short *sampleBuffer;
static bool debug_nn = false; // Set this to true to see e.g. features generated from the raw signal
static int print_results = -(EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW);
static bool sensor_active = false;

void setColor(bool red, bool green, bool blue) {
    digitalWrite(LED_PIN_RED, red);
    digitalWrite(LED_PIN_GREEN, green);
    digitalWrite(LED_PIN_BLUE, blue);
}

/**
 * @brief      Arduino setup function
 */
void setup()
{
    // put your setup code here, to run once:
    Serial.begin(115200);
    // comment out the below line to cancel the wait for USB connection (needed for native USB)
    while (!Serial);
    Serial.println("Edge Impulse Inferencing Demo");

    // summary of inferencing settings (from model_metadata.h)
    // Serial.println("Inferencing settings:");
    // Serial.print("\tInterval: ");
    // Serial.print((float)EI_CLASSIFIER_INTERVAL_MS, 2);
    // Serial.println(" ms.");
    // Serial.print("\tFrame size: ");
    // Serial.println(EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE);
    // Serial.print("\tSample length: ");
    // Serial.print(EI_CLASSIFIER_RAW_SAMPLE_COUNT / 16);
    // Serial.println(" ms.");
    // Serial.print("\tNo. of classes: ");
    // Serial.println(sizeof(ei_classifier_inferencing_categories) / sizeof(ei_classifier_inferencing_categories[0]));

    run_classifier_init();
    if (microphone_inference_start(EI_CLASSIFIER_SLICE_SIZE) == false) {
        Serial.print("ERR: Could not allocate audio buffer (size ");
        Serial.print(EI_CLASSIFIER_RAW_SAMPLE_COUNT);
        Serial.println("), this could be due to the window length of your model");
        return;
    }

    // // Inicijalizacija LED dioda
    // pinMode(LED_PIN_RED, OUTPUT);
    // pinMode(LED_PIN_GREEN, OUTPUT);
    // pinMode(LED_PIN_BLUE, OUTPUT);

    // // Inicijalizacija LED diode
    // pinMode(LED_PIN, OUTPUT);
    // digitalWrite(LED_PIN, LOW); // Osiguraj da je LED inicijalno isključen

    // setColor(LOW, LOW, LOW); // Inicijalno ugašena LED

    // Inicijalizacija senzora
    if (!IMU.begin()) {
        Serial.println("Nije uspelo pokretanje IMU!");
        while (1);
    }

    if (!BARO.begin()) {
        Serial.println("Nije uspelo pokretanje LPS22HB!");
        while (1);
    }

    if (!HTS.begin()) {
        Serial.println("Nije uspelo pokretanje HTS221!");
        while (1);
    }
}

/**
 * @brief      Arduino main function. Runs the inferencing loop.
 */
void loop()
{
    bool m = microphone_inference_record();
    if (!m) {
        Serial.println("ERR: Failed to record audio...\n");
        return;
    }

    signal_t signal;
    signal.total_length = EI_CLASSIFIER_SLICE_SIZE;
    signal.get_data = &microphone_audio_signal_get_data;
    ei_impulse_result_t result = {0};

    EI_IMPULSE_ERROR r = run_classifier_continuous(&signal, &result, debug_nn);
    if (r != EI_IMPULSE_OK) {
        Serial.print("ERR: Failed to run classifier (");
        Serial.print(r);
        Serial.println(")");
        return;
    }

    if (++print_results >= (EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW)) {
        // print the predictions
        // Serial.print("Predictions ");
        // Serial.print("(DSP: ");
        // Serial.print(result.timing.dsp);
        // Serial.print(" ms., Classification: ");
        // Serial.print(result.timing.classification);
        // Serial.print(" ms., Anomaly: ");
        // Serial.print(result.timing.anomaly);
        // Serial.println(" ms.)");
        // Serial.println(":");

        for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
            // Serial.print("    ");
            // Serial.print(result.classification[ix].label);
            // Serial.print(": ");
            // Serial.println(result.classification[ix].value, 5);
        }

#if EI_CLASSIFIER_HAS_ANOMALY == 1
        // Serial.print("    anomaly score: ");
        // Serial.println(result.anomaly, 3);
#endif

        print_results = 0;
    }

    // Provera da li je prepoznata komanda "hello"
    if (result.classification[2].value > 0.8) {
        sensor_active = true; // Aktiviraj senzore
    }

    // Provera da li je prepoznata komanda "stop"
    if (result.classification[3].value > 0.8) {
        sensor_active = false; // Deaktiviraj senzore
    }

    // Ako su senzori aktivni, čitaj njihove vrednosti
    if (sensor_active) {
        float temperature = HTS.readTemperature();
        float humidity = HTS.readHumidity();
        float pressure = BARO.readPressure();

        //Serial.print("Temperatura: ");
        Serial.print(temperature);
        Serial.print(", ");
        //Serial.print("Pritisak: ");
        Serial.print(pressure);
        Serial.print(", ");
        //Serial.print("Vlažnost: ");
        Serial.println(humidity);

        // if (pressure > 100) {
        //     // Akcija: Uključi plavu LED, isključi crvenu i zelenu
        //     setColor(HIGH, LOW, LOW);
        // }
    } else {
        // Ako senzori nisu aktivni, napravi malu pauzu
        // delay(1);
    }
}

/**
 * @brief      PDM buffer full callback
 *             Get data and call audio thread callback
 */
static void pdm_data_ready_inference_callback(void)
{
    int bytesAvailable = PDM.available();

    // read into the sample buffer
    int bytesRead = PDM.read((char *)&sampleBuffer[0], bytesAvailable);

    if (record_ready == true) {
        for (int i = 0; i < bytesRead >> 1; i++) {
            inference.buffers[inference.buf_select][inference.buf_count++] = sampleBuffer[i];

            if (inference.buf_count >= inference.n_samples) {
                inference.buf_select ^= 1;
                inference.buf_count = 0;
                inference.buf_ready = 1;
            }
        }
    }
}

/**
 * @brief      Init inferencing struct and setup/start PDM
 *
 * @param[in]  n_samples  The n samples
 *
 * @return     { description_of_the_return_value }
 */
static bool microphone_inference_start(uint32_t n_samples)
{
    inference.buffers[0] = (signed short *)malloc(n_samples * sizeof(signed short));

    if (inference.buffers[0] == NULL) {
        return false;
    }

    inference.buffers[1] = (signed short *)malloc(n_samples * sizeof(signed short));

    if (inference.buffers[1] == NULL) {
        free(inference.buffers[0]);
        return false;
    }

    sampleBuffer = (signed short *)malloc((n_samples >> 1) * sizeof(signed short));

    if (sampleBuffer == NULL) {
        free(inference.buffers[0]);
        free(inference.buffers[1]);
        return false;
    }

    inference.buf_select = 0;
    inference.buf_count = 0;
    inference.n_samples = n_samples;
    inference.buf_ready = 0;

    // configure the data receive callback
    PDM.onReceive(&pdm_data_ready_inference_callback);

    PDM.setBufferSize((n_samples >> 1) * sizeof(int16_t));

    // initialize PDM with:
    // - one channel (mono mode)
    // - a 16 kHz sample rate
    if (!PDM.begin(1, EI_CLASSIFIER_FREQUENCY)) {
        Serial.println("Failed to start PDM!");
    }

    // set the gain, defaults to 20
    PDM.setGain(127);

    record_ready = true;

    return true;
}

/**
 * @brief      Wait on new data
 *
 * @return     True when finished
 */
static bool microphone_inference_record(void)
{
    bool ret = true;

    if (inference.buf_ready == 1) {
        Serial.println(
            "Error sample buffer overrun. Decrease the number of slices per model window "
            "(EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW)\n");
        ret = false;
    }

    while (inference.buf_ready == 0) {
        delay(1);
    }

    inference.buf_ready = 0;

    return ret;
}

/**
 * Get raw audio signal data
 */
static int microphone_audio_signal_get_data(size_t offset, size_t length, float *out_ptr)
{
    numpy::int16_to_float(&inference.buffers[inference.buf_select ^ 1][offset], out_ptr, length);

    return 0;
}

/**
 * @brief      Stop PDM and release buffers
 */
static void microphone_inference_end(void)
{
    PDM.end();
    free(inference.buffers[0]);
    free(inference.buffers[1]);
    free(sampleBuffer);
}

#if !defined(EI_CLASSIFIER_SENSOR) || EI_CLASSIFIER_SENSOR != EI_CLASSIFIER_SENSOR_MICROPHONE
#error "Invalid model for current sensor."
#endif


