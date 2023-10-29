# (a) Inherent Problems
### IMU 
IMU's can have noise and drift which over time can lead to inaccuracies. Calibration must be set well to have reliable data. Errors finding the first correct orientation and these errors can be more over time. Integration error can occur when getting the position from double integration the acceleration data.
### Sonar/Ultrasonic
Limited range, not good for long distances. Small angle of coverage, so it does not cover a lot of the area. Reflective surfaces can cause multipath interference which could lead to incorrect measurements.
### GPS
Accuracy can very based on satellites, and atmospheric conditions. Different chips have different accuracies and you have to pay a lot more for more accurate chips. Obstructions from the chip to the satellites can reduce performance, such as being indoors. Signals can be jammed in an error to prevent them from working properly.
### Monocular Camera
Hard to determine the scale of the scene. Motion blur and poor lightning can dramatically reduce performance. Hard for feature extraction if you do not having textures in the images. Cannot provide direct depth information.
# (b) Accuracy of ORB SLAM 3 for our solution
For ORBSLAM3 we got it to work with this repository using docker and a D435 camera with Monocular setting: https://github.com/LMWafer/orb-slam-3-ready 

We ran the algorithm in the robotic's room and once in the hallway and the room. The accuracy was decent expect when it cannot find features and goes back to the starting position to start over as you can see from the video where we went into the hallway as well. This video can be found at 4b/DemoOrbSlam3.webm. Screenshots of the plots and software are also here in png's. A smaller room with more features on the walls would improve the results.

# (c) Autonomous Vehicle Sensors cost and links
### Long-Range (1: front)
A ARS 408-21 can be found on AliExpress for around $950 ([AliExpress](https://www.aliexpress.us/item/3256804291881981.html?src=google&aff_fcid=d8d4ba660f694faf847d1ce42cf264b8-1698610733271-04249-UneMJZVf&aff_fsk=UneMJZVf&aff_platform=aaf&sk=UneMJZVf&aff_trace_key=d8d4ba660f694faf847d1ce42cf264b8-1698610733271-04249-UneMJZVf&terminal_id=581132733ebb41908a4061aa789261f2&afSmartRedirect=y&gatewayAdapt=glo2usa#nav-specification)) it's main website with more information can be found at [Conti-Engineering](https://conti-engineering.com/components/ars-408/)
### LIDAR (6 directional or 1 very capable)
Lidar can range from around $30-$50,000+. [$30(Directional)](https://ozrobotics.com/shop/atom-lidar-a-cost-effective-ranging-module-based-on-tof-technology/), [$60(Directional)](https://www.adafruit.com/product/4441), [$150(Directional)](https://www.sparkfun.com/products/14599), [$230(360)](https://www.amazon.com/youyeetoo-Measuring-Distance-Frequency-Compatible/dp/B0B46MG65X?source=ps-sl-shoppingads-lpcontext&ref_=fplfs&psc=1&smid=AIJ7WJJN4HG6E)
Of course you could also go with this one that costs over $50k [AliExpress](https://www.aliexpress.us/item/3256803817977616.html?src=google&aff_fcid=674ca8da553142bcb0bb28ac0b8ffcbe-1698611573188-06157-UneMJZVf&aff_fsk=UneMJZVf&aff_platform=aaf&sk=UneMJZVf&aff_trace_key=674ca8da553142bcb0bb28ac0b8ffcbe-1698611573188-06157-UneMJZVf&terminal_id=581132733ebb41908a4061aa789261f2&afSmartRedirect=y&gatewayAdapt=glo2usa) - [RobosenseAI](https://www.robosense.ai/en/rslidar/RS-Ruby_Plus)
### Camera (4: front, right, left, and back)
For car cameras I saw ranges from $30-350 each. We'll go with the Intel RealSense Depth Camera D435 for $265 [Amazon](https://www.amazon.com/Intel-Realsense-D435-Webcam-FPS/dp/B07BLS5477/ref=sr_1_3?keywords=depth%2Bcamera&qid=1698614271&sr=8-3&ufe=app_do%3Aamzn1.fos.ac2169a1-b668-44b9-8bd0-5ec63b24bcb5&th=1)
### Short/Medium-Range Radar (7: 3 front, 2 blind spot, 2 back (less for better radar))
The SRR 308 is a Short range radar from [Continental](https://conti-engineering.com/components/srr-308/) - [AilExpress](https://www.aliexpress.us/item/3256805685702234.html?src=google&aff_fcid=0d12567a183042b48eba1c9a8338a0aa-1698614698665-04528-UneMJZVf&aff_fsk=UneMJZVf&aff_platform=aaf&sk=UneMJZVf&aff_trace_key=0d12567a183042b48eba1c9a8338a0aa-1698614698665-04528-UneMJZVf&terminal_id=581132733ebb41908a4061aa789261f2&afSmartRedirect=y&gatewayAdapt=glo2usa) for $1300 though probably don't need 7 of these, maybe 4 or less. There were also some multi lane radars that might be able to do it cheaper with fewer cameras.
## (c) Total
We'll say $1000 for Long-Range, LIDAR $150 x 6 = $900, Cameras $265 x 4 = $1,060, and Short/Medium-Range Radar ranges from $1300 - $5200. Totalling for $4260-8160.