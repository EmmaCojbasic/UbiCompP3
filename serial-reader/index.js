const SerialPort = require('serialport');
const mqtt = require('mqtt');

const parsers = SerialPort.parsers;
const parser = new parsers.Readline({
    delimiter:'\r\n'
});

const qos = 2;
const mqttAddress = 'tcp://localhost:1883'
const clientId = 'serial-reader'
const username = 'serial-reader'
const password = 'serial-reader'
const topic = 'serial-reader/data'

const mqttClient = mqtt.connect(mqttAddress, {
    clientId,
    connectTimeout: 4000,
    username: username,
    password: password,
    reconnectPeriod: 1000
})

const port = new SerialPort('COM7', { 
    baudRate: 115200,
    dataBits: 8,
    parity: 'none',
    stopBits: 1,
    flowControl: false
});

port.pipe(parser);

port.on('open', function () {
    port.set({ dtr: true, rts: true });
})

parser.on('data', function(data){
    console.log(data);
    const values = data.split(", ").map(val => parseFloat(val.trim()));
    const jsonObject = {
        temperature: values[0],
        pressure: values[1],
        humidity: values[2]
    }
    mqttClient.publish(topic, JSON.stringify(jsonObject), { qos }, error => {
        if (error) {
            console.error('ERROR: ', error)
        }
    })
});