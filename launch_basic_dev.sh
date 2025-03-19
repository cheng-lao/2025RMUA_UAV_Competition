
catkin_make
roscore &
source ./devel/setup.bash
source ./devel/setup.bash
source ./devel/setup.bash
# Launch the basic_dev node


cd ./src/depth_stereo/src
python main.py --restore_ckpt disparity_last.ckpt --use_model  --command_model_path command_model_epoch116.pth &
PYTHON_PID="$!"
echo "PYTHON_PID IS " $PYTHON_PID
sleep 5

cd -
roslaunch ./basic_dev.launch  &
ROS_PID="$!"
echo "ROS_PID IS " $ROS_PID
sleep 90


if [ $ROS_PID ]
then
  echo "Killing simulator with PID $ROS_PID ----AAAAA"
  kill -SIGINT "$ROS_PID"
fi

if [ $PYTHON_PID ]
then
  echo "Killing python with PID $PYTHON_PID ----AAAAA"
  kill -SIGINT "$PYTHON_PID"
fi
