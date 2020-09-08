 #include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <MQTTClient.h>
#include <time.h>
#define ADDRESS     "tcp://localhost:1883"
//#define PAYLOAD     "Hello World!"//
#define QOS         2
#define QOS1        1
#define TIMEOUT     10000L
volatile MQTTClient_deliveryToken deliveredtoken;
char TOPIC[1000];
char CLIENTID[100];
void delivered(void *context, MQTTClient_deliveryToken dt)
{
    printf("Message with token value %d delivery confirmed\n", dt);
    printf("\n");
    deliveredtoken = dt;
}
int msgarrvd(void *context, char *topicName, int topicLen, MQTTClient_message *message)
{
    int i;
    char* payloadptr;
    printf("Message arrived\n");
    printf("     topic: %s\n", topicName);
    printf("   message: ");
    payloadptr = (char*) message->payload;
    for(i=0; i<message->payloadlen; i++)
    {
        putchar(*payloadptr++);
    }
    putchar('\n');
    MQTTClient_freeMessage(&message);
    MQTTClient_free(topicName);
    return 1;
}
void connlost(void *context, char *cause)
{
    printf("\nConnection lost\n");
    printf("     cause: %s\n", cause);
}
int main(int argc, char* argv[])
{
    printf("enter person name who wants to publish");
    scanf("%s",CLIENTID);
    printf("\n");
    printf("enter person number you want to send the message to:");
    scanf("%s",TOPIC);
    MQTTClient client;
    MQTTClient_willOptions will_opts = MQTTClient_willOptions_initializer;
    MQTTClient_connectOptions conn_opts = MQTTClient_connectOptions_initializer;
    MQTTClient_message pubmsg = MQTTClient_message_initializer;
    MQTTClient_deliveryToken token;
    conn_opts.keepAliveInterval = 20;
    conn_opts.cleansession = 0;
    int rc;
    conn_opts.will = &will_opts;
    will_opts.topicName =TOPIC;
    will_opts.message ="LEFT GROUP";
    MQTTClient_create(&client, ADDRESS, CLIENTID,
        MQTTCLIENT_PERSISTENCE_NONE, NULL);
    MQTTClient_setCallbacks(client, NULL, connlost, msgarrvd, delivered);
    if ((rc = MQTTClient_connect(client, &conn_opts)) != MQTTCLIENT_SUCCESS)
    {
        printf("Failed to connect, return code %d\n", rc);
        exit(EXIT_FAILURE);
    }
    char PAYLOAD[10000];
    scanf("%s",PAYLOAD);

    pubmsg.payload = (void*) PAYLOAD;
    pubmsg.payloadlen = strlen(PAYLOAD);
    pubmsg.qos = QOS;
    pubmsg.retained = 0;
    deliveredtoken = 0;

    MQTTClient_publishMessage(client, TOPIC, &pubmsg, &token);
     printf("Waiting for publication of %s\n"
            "on topic %s for client with ClientID: %s\n",
            PAYLOAD, TOPIC, CLIENTID);
    while(deliveredtoken != token);
    MQTTClient_disconnect(client, 10000);
    MQTTClient_destroy(&client);
    return rc;
}


