const mqtt = require('mqtt');
const { InfluxDB, Point } = require('@influxdata/influxdb-client');
const tf = require('tensorflow/lite');
const io = require('socket.io-client');

const mqttAddress = 'tcp://emqx:1883'
const clientId = 'iot-application'
const username = 'iot-application'
const password = 'iot-application'
const topic = 'serial-reader/data'

const url = 'http://influxdb:8086';
const token = 'brZmV0HPOUTQhuKpuVDyRIv2u7EDo18748lI8nRASiJlcg69lz0hkwupXop_jTm1EpcZEOWUqWzhwW9lCrpARQ=='; 
const organization = 'elfak';
const bucket = 'sr3';

const influxdb = new InfluxDB({ url, token });

const writeApi = influxdb.getWriteApi(organization, bucket);

const socket = io('http://127.0.0.1:5500/');  //live server vscode

const mqttClient = mqtt.connect(mqttAddress, {
    clientId,
    connectTimeout: 4000,
    username: username,
    password: password,
    reconnectPeriod: 1000
})

mqttClient.on('connect', () => {
    console.log('Connected')
    mqttClient.subscribe(topic, () => {
        console.log(`Subscribed to topic '${topic}'`)
    })
})

const path = require('path');
const modelPath = path.join(__dirname, '..', 'model', 'env_model.tflite');
const model = new tf.Model(modelPath);

const runModel = (data) => {
  const input = new Float32Array([data.temperature, data.pressure, data.humidity]);
  const output = new Float32Array(1); 

  model.predict(input, output);

  return output[0]; 
};

mqttClient.on('message', (topic, payload) => {
    const json = JSON.parse(payload)
    console.log(`Received Message from '${topic}': `, json)
    
    const point = new Point('sensor-measurement')
      .floatField('temperature', json.temperature)
      .floatField('pressure', json.pressure)
      .floatField('humidity', json.pressure)
      .timestamp(new Date());
  
    writeApi.writePoint(point);
    writeApi.flush();

    // poziv ML
    const prediction = runModel(json);

    // emit na socket
    if (prediction >= 0.5) { // veća verovatnoća da je dobro okruženje (binarni klasifikator)
      socket.emit('environment_status', { status: 'good', data: json });
      console.log('Emitting good environment status to web app');
    } else {
      socket.emit('environment_status', { status: 'bad', data: json });
      console.log('Emitting bad environment status to web app');
    }

})

process.on('exit', () => {
  writeApi.close().then(() => {
    console.log('WRITE API closed');
  });
});