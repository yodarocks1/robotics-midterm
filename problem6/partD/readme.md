# CS5510 Midterm Part 6d
## Front of house robot
We are trying to design a front of house assistant robot that can take customer orders and handle costumer requests in a resraunt like taco bell. 

## Our stack:

Voice recognition (to change voice input to text)
Our VER script. (to detect human expression)
Our NLP reader. (to detect emotion in text)
A Chatbot service to respond (to detect intent and respnd to reply)
A messeging service. (to fetch human/manger in the event it is needed)

This would use our Two scripts to detect how the user is feeling, in the event any request is made to see a manger or human or frustration with the machine is expressed as a combination of words and facial expression than a message will be sent to the manger using the messaging service. 

The robot could also use the data to change how it response, if a user seems angry or sad it could offer free food or corrections similar to how places do now. 

see the UML in UML.png
see the flow chart in Flow.png
