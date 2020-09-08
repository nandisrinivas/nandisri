To run one-one chat:
0)Run hive mqtt
1)pub.c for publisher code
2) sub.c for subscriber code
3)To run publisher
$gcc pub.c -l paho-mqtt3c -o pub.out
$./pub.out

4)To run subscriber
$gcc sub.c -l paho-mqtt3c -o sub.out
$./sub.out

To run group chat:
$gcc subgroup.c -l paho-mqtt3c -o subgroup.out
$./subgroup.out
$gcc pubgroup.c -l paho-mqtt3c -o pubgroup.out
$./pubgroup.out
NOTE: Clientid should be unique for each subscriber as well as publisher.
