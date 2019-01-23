"""
Tests that can be applied to real test-flights as well
as simulated test-flights
"""

from uloganalysis import attitudeanalysis as attanl
from uloganalysis import positionanalysis as posanl
from uloganalysis import ulogconv
from uloganalysis import loginfo
import pyulog
import os
import numpy as np
import pytest


def setup_dataframe(self, filepath, topics, COLUMNS_ZERO_ORDER_HOLD):
    # Check if any of the topics exist in the topics exists in the log file
    try:
        self.ulog = pyulog.ULog(filepath, topics)
    except:
        print("Not a single topic that is needed for this test exists in the provided ulog file. Abort test")
        assert False

    # Check for every topic separately if it exists in the log file
    for i in range(len(self.ulog.data_list)):
        if self.ulog.data_list[i].name in topics:
            idx = topics.index(self.ulog.data_list[i].name)
            topics.pop(idx)

    if len(topics) > 0:
        print("\033[93m" + "The following topics do not exist in the provided ulog file: " + "\033[0m")
        print(topics)
        pytest.skip("Skip this test because topics are missing")
    else:
        self.df = ulogconv.merge(ulogconv.createPandaDict(self.ulog), COLUMNS_ZERO_ORDER_HOLD)
    
    # return self


class TestAttitude:
    """
    Test Attitude related constraints
    """
    def test_tilt_desired(self, filepath):

        topics = [
            "vehicle_attitude",
            "vehicle_attitude_setpoint",
            "vehicle_status",
        ]

        COLUMNS_ZERO_ORDER_HOLD = []

        setup_dataframe(self, filepath, topics, COLUMNS_ZERO_ORDER_HOLD)
    
        # During Manual / Stabilized and Altitude, the tilt threshdol should not exceed
        # MPC_MAN_TILT_MAX

        attanl.add_desired_tilt(self.df)
        man_tilt = (
            loginfo.get_param(self.ulog, "MPC_MAN_TILT_MAX", 0) * np.pi / 180
        )
        assert self.df[
            (
                (self.df.T_vehicle_status_0__F_nav_state == 0)
                | (self.df.T_vehicle_status_0__F_nav_state == 1)
            )
            & (
                self.df.T_vehicle_attitude_setpoint_0__NF_tilt_desired
                > man_tilt
            )
        ].empty


class TestRTLHeight:
    # The return to home height changes with the distance from home
    # mode was triggered
    # check the height above ground while the drone returns to home. compare it with 
    # the allowed maximum or minimum heights, until the drone has reached home and motors have been turned off

    def test_rtl(self, filepath):

        topics = [
            "vehicle_local_position",
            "vehicle_status",
        ]

        COLUMNS_ZERO_ORDER_HOLD = []

        setup_dataframe(self, filepath, topics, COLUMNS_ZERO_ORDER_HOLD)

        # drone parameters: below rtl_min_dist, the drone follows different rules than outside of it.
        rtl_min_dist = (
            loginfo.get_param(self.ulog, "RTL_MIN_DIST", 0)
        )
        rtl_return_alt = (
            loginfo.get_param(self.ulog, "RTL_RETURN_ALT", 0)
        )
        rtl_cone_dist = (
            loginfo.get_param(self.ulog, "RTL_CONE_DIST", 0)
        )

        NAVIGATION_STATE_AUTO_RTL = 5           # see https://github.com/PX4/Firmware/blob/master/msg/vehicle_status.msg
        thresh = 1                              # Threshold for position inaccuracies, in meters

        posanl.add_horizontal_distance(self.df)

        # Run the test every time that RTL was triggered
        self.df['T_vehicle_status_0__F_nav_state_group2'] = (self.df.T_vehicle_status_0__F_nav_state != self.df.T_vehicle_status_0__F_nav_state.shift()).cumsum()
        state_group = self.df.groupby(['T_vehicle_status_0__F_nav_state_group2'])
        for g, d in state_group:
            # Check that RTL was actually triggered 
            # at least two consecutive T_vehicle_status_0__F_nav_state values have to 
            # be equal to NAVIGATION_STATE_AUTO_RTL in order to confirm that RTL has been triggered
            if d.T_vehicle_status_0__F_nav_state.count() > 1 and d.T_vehicle_status_0__F_nav_state[0] == NAVIGATION_STATE_AUTO_RTL:
                height_at_RTL = abs(d.T_vehicle_local_position_0__F_z[0])
                distance_at_RTL = d.T_vehicle_local_position_0__NF_abs_horizontal_dist[0]
                max_height_during_RTL = abs(max(d.T_vehicle_local_position_0__F_z))

                if rtl_cone_dist > 0:
                    # Drone should not rise higher than height defined by a cone (definition taken from rtl.cpp file in firmware)
                    max_height_within_RTL_MIN_DIST = 2 * distance_at_RTL
                else:
                    # If no cone is defined, drone should not rise at all within certain radius around home
                    max_height_within_RTL_MIN_DIST = height_at_RTL
                
                # check if a value of the z position after triggering RTL is larger than allowed value
                if (distance_at_RTL < rtl_min_dist) & (height_at_RTL < max_height_within_RTL_MIN_DIST):
                    assert max_height_during_RTL < max_height_within_RTL_MIN_DIST + thresh

                elif (distance_at_RTL < rtl_min_dist) & (height_at_RTL >= max_height_within_RTL_MIN_DIST): 
                    assert max_height_during_RTL < height_at_RTL + thresh

                elif (distance_at_RTL >= rtl_min_dist) & (height_at_RTL < rtl_return_alt): 
                    assert max_height_during_RTL < rtl_return_alt + thresh

                elif (distance_at_RTL >= rtl_min_dist) & (height_at_RTL > rtl_return_alt): 
                    assert max_height_during_RTL < height_at_RTL + thresh




class TestAvoidance:
    """
    Test Avoidance related constraints
    """
    def test_no_detection_no_movement(self, filepath):
        """
        As long as the drone does not detect anything, it should not make any large movements while there is no stick input.
        """

        topics = [
            "vehicle_status",
            "distance_sensor", # 
            "manual_control_setpoint", #obsavoid_switch, 
        ]

        COLUMNS_ZERO_ORDER_HOLD = []

        setup_dataframe(self, filepath, topics, COLUMNS_ZERO_ORDER_HOLD)

        NAVIGATION_STATE_POSCTL = 2 # TODO get that param directly from the log file??
        MINIMUM_INCREMENT = 5
        MPC_HOLD_DZ = 0.1

        # check if vehicle_status is equal to NAVIGATION_STATE_POSCTL. extract all rows for which that is valid
        # check if obsavoid_switch is turned on (turned on means values 1 and 2. value 3 means turned off).  extract all rows for which that is valid
        # check the manual control setpoints for x and y. They have to be close to zero, thus they have to be smaller than MPC_HOLD_DZ = 0.1
        self.df["avoidance_test_conditions"] = \
        (
            (self.df["T_vehicle_status_0__F_nav_state"] == NAVIGATION_STATE_POSCTL) & \
            (self.df["T_manual_control_setpoint_0__F_obsavoid_switch"] == 1) & \
            (self.df["T_manual_control_setpoint_0__F_obsavoid_switch"] == 2) & \
            (self.df["T_manual_control_setpoint_0__F_x"].abs() < MPC_HOLD_DZ) & \
            (self.df["T_manual_control_setpoint_0__F_y"].abs() < MPC_HOLD_DZ)
        )

        # check the distance sensor. No obstacle: distance_sensor.current_distance is equal to MINIMUM_INCREMENT = 5
        # find all entries where the distance sensor indicates that there is no object in front of the drone.  
        self.df["no_obstacle"] = (self.df["T_distance_sensor_0__F_current_distance"] != MINIMUM_INCREMENT)
        # print(self.df["T_distance_sensor_0__F_current_distance"])

        # extract the position setpoints in x and y direction: vehicle_local_position_setpoint_0.x and .y
        self.df = self.df.drop(self.df[self.df["avoidance_test_conditions"] == True].index)

        # group the remaining entries. Every group has to contain more than a certain number of values in order to be used in the test
        state_group = self.df.groupby(["no_obstacle"])

        # for g, d in state_group:
        #     if d.len

        # print(self.df["avoidance_test_conditions"])


        assert True



# class TestSomething:
#
    # def test_1(self, filepath):
    #     topics = [
    #         "topic1",
    #         "topic2",
    #     ]
    #    setup_dataframe(self, filepath)
#        assert True
#    def test_2(self, filepath):
    #     topics = [
    #         "topic1",
    #         "topic2",
    #     ]
        # setup_dataframe(self, filepath)
#        assert True