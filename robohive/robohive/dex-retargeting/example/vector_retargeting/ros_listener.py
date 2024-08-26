import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray 
def pose_callback(msg):
    # Print the received PoseStamped data
    rospy.loginfo(f"Received PoseStamped data:\n"
                  f"Data: {msg.data}\n")

def listener():
    # Initialize the ROS node
    rospy.init_node('pose_listener', anonymous=True)

    # Subscribe to the /natnet_ros/Test/pose topic
    rospy.Subscriber("/manus_glove_data_right", Float32MultiArray, pose_callback)

    # Keep the node running until it's manually stopped
    rospy.spin()

if __name__ == '__main__':
    listener()
