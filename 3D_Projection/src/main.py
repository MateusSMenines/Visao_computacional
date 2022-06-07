import sys
import numpy as np
import matplotlib.pyplot as plt
import os
from PyQt5 import QtGui
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QLineEdit, QSpinBox
from PyQt5.QtGui import QPainter
from PyQt5.QtCore import *  
from math import pi,cos,sin


class App(QMainWindow):

    def __init__(self):
        super().__init__()

        self.a = 100
        self.w = 100
        self.largura = 900
        self.altura = 600
        self.title = "Trabalho 1"

        self.MyUI()


    def MyUI(self):

        self.label()
        self.text_label()
        self.camera_parameter()
        self.button()
        self.spinbox()
        self.object() 
        self.image() 
        self.projection()
        self.plot3d() 
        self.plot2d()
        self.loading_window()


    def rotation(self,axis=None,x=0,y=0,z=0):


        x = x*pi/180
        y = y*pi/180
        z = z*pi/180

        if axis == 'z':
            rotation_matrix=np.array([[cos(z),-sin(z),0,0],[sin(z),cos(z),0,0],[0,0,1,0],[0,0,0,1]])
            return rotation_matrix
        elif axis == 'x':
            rotation_matrix=np.array([[1,0,0,0],[0, cos(x),-sin(x),0],[0, sin(x), cos(x),0],[0,0,0,1]])
            return rotation_matrix
        elif axis == 'y':
            rotation_matrix=np.array([[cos(y),0, sin(y),0],[0,1,0,0],[-sin(y), 0, cos(y),0],[0,0,0,1]])
        else:
            rotation_matrix=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
            
        return rotation_matrix


    def translation(self,dx=0,dy=0,dz=0):

        T = np.eye(4)
        T[0,-1]=dx
        T[1,-1]=dy
        T[2,-1]=dz

        return T
        

    def plot3d(self):
        
        def draw_arrows(point,base,axis,length=0.3):
            
            # Plot vector of x-axis
            axis.quiver(point[0],point[1],point[2],base[0,0],base[1,0],base[2,0],color='red',pivot='tail',  length=length)
            # Plot vector of y-axis
            axis.quiver(point[0],point[1],point[2],base[0,1],base[1,1],base[2,1],color='green',pivot='tail',  length=length)
            # Plot vector of z-axis
            axis.quiver(point[0],point[1],point[2],base[0,2],base[1,2],base[2,2],color='blue',pivot='tail',  length=length)

            return axis 

        def axisEqual3D(ax):
            
            extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
            sz = extents[:,1] - extents[:,0]
            centers = np.mean(extents, axis=1)
            maxsize = max(abs(sz))
            r = maxsize/2
            for ctr, dim in zip(centers, 'xyz'):
                getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)

        ax0 = plt.axes(projection='3d')
        ax0.plot3D(self.house[0,:], self.house[1,:], self.house[2,:], 'red')
        ax0.plot3D(self.door[0,:], self.door[1,:], self.door[2,:], 'blue')
        ax0.plot3D(self.window[0,:], self.window[1,:], self.window[2,:], 'blue')
        ax0.plot3D(self.camera[0,:], self.camera[1,:], self.camera[2,:], 'black')
        ax0 = draw_arrows(self.point_camera,self.base_camera,ax0,length=0.1)
        ax0 = draw_arrows(self.point_world,self.base_world,ax0,length=0.1)
        axisEqual3D(ax0)
        ax0.view_init(elev=25,azim=-40)
        ax0.dist=7
        ax0.figure.savefig('image.png',dpi = 75, bbox_inches = 'tight')
        plt.close()
        
        self.image1.setPixmap(QtGui.QPixmap('image.png'))
        self.image1.adjustSize()

        self.label_position_plot.setText(f" X = {self.point_camera[0][0]:.2f}, Y = {self.point_camera[1][0]:.2f}, Z = {self.point_camera[2][0]:.2f}")
        self.label_position_plot.adjustSize()
        os.remove("image.png")


    def plot2d(self):

        ax0 = plt.axes()
        ax0.plot(self.camera_house[0,:], self.camera_house[1,:], 'red')
        ax0.plot(self.camera_door[0,:], self.camera_door[1,:], 'blue')
        ax0.plot(self.camera_window[0,:], self.camera_window[1,:], 'blue')
        ax0.set_xlim(0,640)
        ax0.set_ylim(0,480)
        ax0.dist=8
        ax0.grid(bool)
        plt.gca().invert_yaxis()

        ax0.figure.savefig('image1.png',dpi = 78, bbox_inches = 'tight')
        plt.close()

        self.image2.setPixmap(QtGui.QPixmap('image1.png'))
        self.image2.adjustSize()
        os.remove("image1.png")


    def transformation(self,axis=None,reference=None,x=0,y=0,z=0,r=0,p=0,ya=0):

        if reference == "world":

            T = self.translation(x/100, y/100, z/100)
            R = self.rotation(axis, r, p, ya)
            M = np.dot(T,R)
            M_acu = np.dot(M, self.M_camera)

            camera = np.dot(M, self.camera)
            point_camera = np.dot(M, self.point_camera)
            base_camera = np.dot(M, self.base_camera)

        elif reference == "camera":

            M_inv = np.linalg.inv(self.M_camera)
            T = self.translation(x/100, y/100, z/100)
            R = self.rotation(axis, r, p, ya)
            M = np.dot(T,R)
            M = np.linalg.multi_dot([self.M_camera, M, M_inv])
            M_acu = np.dot(M, self.M_camera)
            
            camera = np.dot(M, self.camera)
            point_camera = np.dot(M, self.point_camera)
            base_camera = np.dot(M, self.base_camera)

        self.M_camera = M_acu
        self.camera = camera
        self.point_camera = point_camera
        self.base_camera = base_camera

        self.plot3d()
        self.projection()


    def projection(self):


        def change_world2cam (M,point_world):
            M_inv = np.linalg.inv(M)
            p_cam = np.dot(M_inv,point_world)

            return p_cam


        def point_project(M_canonical, P_2d, M_intrinsic):

            P_2d = np.dot(M_canonical, P_2d)

            for i in range(len(P_2d)):
            
                P_2d[i] = P_2d[i]/P_2d[2]

            P_cam = np.dot(M_intrinsic, P_2d)

            return P_cam


        point_house_camera =  change_world2cam(self.M_camera, self.house)
        point_door_camera = change_world2cam(self.M_camera, self.door)
        point_window_camera = change_world2cam(self.M_camera, self.window)

        M_canonical = np.array([[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 1, 0]])
        M_intrinsic = np.array([[self.f/100*self.sx, self.s0, self.ox],[0, self.f/100*self.sy, self.oy],[0, 0, 1]])

        self.camera_house = point_project(M_canonical, point_house_camera, M_intrinsic)          
        self.camera_door = point_project(M_canonical, point_door_camera, M_intrinsic)
        self.camera_window = point_project(M_canonical, point_window_camera, M_intrinsic)

        self.plot2d()


    def label(self):


        label_world = QLabel(self)
        label_world.setText("World reference:")
        label_world.move(5,3)
        label_world.adjustSize()

        label_camera = QLabel(self)
        label_camera.setText("Camera reference:")
        label_camera.move(305,3)
        label_camera.adjustSize()

        label_parameter = QLabel(self)
        label_parameter.setText("Camera parameter:")
        label_parameter.move(605,3)
        label_parameter.adjustSize()

        label_plot_world = QLabel(self)
        label_plot_world.setText("Ploting image world")
        label_plot_world.move(150,178)
        label_plot_world.adjustSize()

        label_plot_camera = QLabel(self)
        label_plot_camera.setText("Ploting image camera")
        label_plot_camera.move(620,178)
        label_plot_camera.adjustSize()

        label_plot_uni1 = QLabel(self)
        label_plot_uni1.setText("cm")
        label_plot_uni1.move(80,43)
        label_plot_uni1.adjustSize()

        label_plot_uni2 = QLabel(self)
        label_plot_uni2.setText("cm")
        label_plot_uni2.move(80,93)
        label_plot_uni2.adjustSize()

        label_plot_uni3 = QLabel(self)
        label_plot_uni3.setText("cm")
        label_plot_uni3.move(80,143)
        label_plot_uni3.adjustSize()

        label_plot_uni4 = QLabel(self)
        label_plot_uni4.setText("degree")
        label_plot_uni4.move(240,43)
        label_plot_uni4.adjustSize()

        label_plot_uni5 = QLabel(self)
        label_plot_uni5.setText("degree")
        label_plot_uni5.move(240,93)
        label_plot_uni5.adjustSize()

        label_plot_uni6 = QLabel(self)
        label_plot_uni6.setText("degree")
        label_plot_uni6.move(240,143)
        label_plot_uni6.adjustSize()

        label_plot_uni7 = QLabel(self)
        label_plot_uni7.setText("cm")
        label_plot_uni7.move(380,43)
        label_plot_uni7.adjustSize()

        label_plot_uni8 = QLabel(self)
        label_plot_uni8.setText("cm")
        label_plot_uni8.move(380,93)
        label_plot_uni8.adjustSize()

        label_plot_uni9 = QLabel(self)
        label_plot_uni9.setText("cm")
        label_plot_uni9.move(380,143)
        label_plot_uni9.adjustSize()

        label_plot_uni10 = QLabel(self)
        label_plot_uni10.setText("degree")
        label_plot_uni10.move(540,43)
        label_plot_uni10.adjustSize()

        label_plot_uni11 = QLabel(self)
        label_plot_uni11.setText("degree")
        label_plot_uni11.move(540,93)
        label_plot_uni11.adjustSize()

        label_plot_uni12 = QLabel(self)
        label_plot_uni12.setText("degree")
        label_plot_uni12.move(540,143)
        label_plot_uni12.adjustSize()

        label_plot_position = QLabel(self)
        label_plot_position.setText("Position camera:")
        label_plot_position.move(13,200)
        label_plot_position.adjustSize()

        label_plot_f = QLabel(self)
        label_plot_f.setText("f(0, 1000)")
        label_plot_f.move(662,42)
        label_plot_f.adjustSize()

        label_plot_sx = QLabel(self)
        label_plot_sx.setText("sx(0, 1000)")
        label_plot_sx.move(662,92)
        label_plot_sx.adjustSize()

        label_plot_sy = QLabel(self)
        label_plot_sy.setText("sy(0, 1000)")
        label_plot_sy.move(662,142)
        label_plot_sy.adjustSize()

        label_plot_s0 = QLabel(self)
        label_plot_s0.setText("s0(-500, 500)")
        label_plot_s0.move(808,42)
        label_plot_s0.adjustSize()

        label_plot_ox = QLabel(self)
        label_plot_ox.setText("ox(0, 500)")
        label_plot_ox.move(808,92)
        label_plot_ox.adjustSize()

        label_plot_oy = QLabel(self)
        label_plot_oy.setText("oy(0, 500)")
        label_plot_oy.move(808,142)
        label_plot_oy.adjustSize()

        self.label_position_plot = QLabel(self)
        self.label_position_plot.move(125,200)


    def text_label(self):


        # text label x location
        self.text_xw = QLineEdit(self)
        self.text_xw.move(35,40)
        self.text_xw.resize(40,25)

        # text label y location
        self.text_yw = QLineEdit(self)
        self.text_yw.move(35,90)
        self.text_yw.resize(40,25)

        # text label z location
        self.text_zw = QLineEdit(self)
        self.text_zw.move(35,140)
        self.text_zw.resize(40,25)
    
        # text label Roll  world location
        self.text_rw = QLineEdit(self)
        self.text_rw.move(195,40)
        self.text_rw.resize(40,25)

        # text label Pit world location
        self.text_pw = QLineEdit(self)
        self.text_pw.move(195,90)
        self.text_pw.resize(40,25)

        # text label Yaw world location
        self.text_yaw = QLineEdit(self)
        self.text_yaw.move(195,140)
        self.text_yaw.resize(40,25)

        # text label x camera location
        self.text_xc = QLineEdit(self)
        self.text_xc.move(335,40)
        self.text_xc.resize(40,25)

        # text label y camera location
        self.text_yc = QLineEdit(self)
        self.text_yc.move(335,90)
        self.text_yc.resize(40,25)

        # text label z camera location
        self.text_zc = QLineEdit(self)
        self.text_zc.move(335,140)
        self.text_zc.resize(40,25)

        # text label Roll camera location
        self.text_rc = QLineEdit(self)
        self.text_rc.move(495,40)
        self.text_rc.resize(40,25)

        # text label Pit camera location
        self.text_pc = QLineEdit(self)
        self.text_pc.move(495,90)
        self.text_pc.resize(40,25)

        # text label Yaw camera location
        self.text_yac = QLineEdit(self)
        self.text_yac.move(495,140)
        self.text_yac.resize(40,25)
     

    def paintEvent(self, event):

        def rect_1():

            painter = QPainter(self)
            painter.setPen(Qt.black)
            painter.drawRect(2,2,298,173)
        
        def rect_2():
            
            painter = QPainter(self)
            painter.setPen(Qt.black)
            painter.drawRect(300,2,300,173)
        
        def rect_3():
            
            painter = QPainter(self)
            painter.setPen(Qt.black)
            painter.drawRect(600,2,298,173)
        
        def line():

            painter = QPainter(self)
            painter.setPen(Qt.black)
            painter.drawLine(450,175,450,600)
        

        rect_1()
        rect_2()
        rect_3()
        line()


    def object(self):


        def homogeneos(object):

            num_columns = np.size(object,1)
            ones_line = np.ones(num_columns)
            object = np.vstack([object, ones_line])

            return object

        house = np.array([[ 0.24,   0.2,    0.   ],
                          [ 0.24,  -0.2,    0.   ],
                          [ 0.24,  -0.2,    0.48 ],
                          [ 0.24,  -0.216,  0.46 ],
                          [ 0.24,   0,      0.64 ],
                          [ 0.24,   0.2,    0.48 ],
                          [ 0.24,   0.22,   0.456],
                          [ 0.24,   0.2,    0.48 ],
                          [ 0.24,   0.2,    0.   ],
                          [-0.24,   0.2,    0.   ],
                          [-0.24,   0,      0.   ],
                          [-0.24,  -0.2,    0.   ],
                          [ 0.24,  -0.2,    0.   ],
                          [ 0.24,  -0.2,    0.   ],
                          [ 0.24,  -0.2,    0.48 ],
                          [-0.24,  -0.2,    0.48 ],
                          [-0.24,   0.2,    0.48 ],
                          [ 0.24,   0.2,    0.48 ],
                          [ 0.24,  -0.2,    0.48 ],
                          [ 0.24,  -0.22,   0.456],
                          [-0.24,  -0.22,   0.456],
                          [-0.24,  -0.2,    0.48 ],
                          [-0.24,   0,      0.64 ],
                          [ 0.24,   0,      0.64 ],
                          [ 0.24,   0.22,   0.456],
                          [-0.24,   0.22,   0.456],
                          [-0.24,   0.2,    0.48 ],
                          [-0.24,   0,      0.64 ],
                          [-0.24,  -0.2,    0.48 ],
                          [-0.24,  -0.2,    0.   ],
                          [-0.24,   0,      0.   ],
                          [-0.24,   0.2,    0.   ],
                          [-0.24,   0.2,    0.48 ],
                          [-0.24,   0.2,    0.   ]])

        door = np.array([[-0.08, -0.2,   0.  ],
                         [-0.08, -0.2,   0.32],
                         [ 0.08, -0.2,   0.32],
                         [ 0.08, -0.2,   0.  ]])

        window = np.array([[ 0.24, -0.04,  0.32],
                          [ 0.24,  0.04,  0.32],
                          [ 0.24,  0.04,  0.16],
                          [ 0.24, -0.04,  0.16],
                          [ 0.24, -0.04,  0.32]])

        camera = np.array([[0.1, -0.1, 0],
                          [-0.1, -0.1, 0],
                          [-0.1, 0.1,  0],
                          [ 0.1, 0.1,  0],
                          [0.1, -0.1, 0]])

        self.point_world = np.array([[0],[0],[0],[1]])
        e1 = np.array([[1],[0],[0],[0]]) # X
        e2 = np.array([[0],[1],[0],[0]]) # Y
        e3 = np.array([[0],[0],[1],[0]]) # Z
        self.base_world = np.hstack((e1,e2,e3))

        camera = np.transpose(camera)
        house = np.transpose(house)
        door = np.transpose(door)
        window = np.transpose(window)
        
        self.plot_camera = homogeneos(camera)       
        self.house = homogeneos(house)
        self.door = homogeneos(door)
        self.window = homogeneos(window)

        T = self.translation(0, -1.2, 0.3)
        R = self.rotation("x", -90, 0, 0)
        M_camera = np.dot(T,R)

        self.M_camera = M_camera
        self.camera = np.dot(self.M_camera, self.plot_camera)
        self.point_camera = np.dot(self.M_camera, self.point_world)
        self.base_camera = np.dot(self.M_camera, self.base_world)

        self.label_position_plot.setText(f" X = {self.point_camera[0][0]:.2f}, Y = {self.point_camera[1][0]:.2f}, Z = {self.point_camera[2][0]:.2f}")
        self.label_position_plot.adjustSize()


    def image(self):


        self.image1 = QLabel(self)
        self.image1.move(13,230)

        self.image2 = QLabel(self)
        self.image2.move(463,230)


    def camera_parameter(self):


        self.f = 100
        self.sx = 640
        self.sy = 480
        self.s0 = 0
        self.ox = self.sx/2
        self.oy = self.sy/2


    def button(self):
        

        # World (x,y,z) ---------------------------------------------------------------------------------------------

        # button X world location
        self.buttom_xw1 = QPushButton("X++", self)
        self.buttom_xw1.move(5,30)
        self.buttom_xw1.resize(25,20)
        self.buttom_xw1.setStyleSheet('QPushButton {font-size:10px}')
        self.buttom_xw1.clicked.connect(lambda: self.button_click("x world"))

        self.buttom_xw2 = QPushButton("X--", self)
        self.buttom_xw2.move(5,52)
        self.buttom_xw2.resize(25,20)
        self.buttom_xw2.setStyleSheet('QPushButton {font-size:10px}')
        self.buttom_xw2.clicked.connect(lambda: self.button_click("-x world"))

        # button Y world location
        self.buttom_yw1 = QPushButton("Y++", self)
        self.buttom_yw1.move(5,80)
        self.buttom_yw1.resize(25,20)
        self.buttom_yw1.setStyleSheet('QPushButton {font-size:10px}')
        self.buttom_yw1.clicked.connect(lambda: self.button_click("y world"))

        self.buttom_yw2 = QPushButton("Y--", self)
        self.buttom_yw2.move(5,102)
        self.buttom_yw2.resize(25,20)
        self.buttom_yw2.setStyleSheet('QPushButton {font-size:10px}')
        self.buttom_yw2.clicked.connect(lambda: self.button_click("-y world"))

        # button Z world location
        self.buttom_zw1 = QPushButton("Z++", self)
        self.buttom_zw1.move(5,130)
        self.buttom_zw1.resize(25,20)
        self.buttom_zw1.setStyleSheet('QPushButton {font-size:10px}')
        self.buttom_zw1.clicked.connect(lambda: self.button_click("z world"))

        self.buttom_zw2 = QPushButton("Z--", self)
        self.buttom_zw2.move(5,152)
        self.buttom_zw2.resize(25,20)
        self.buttom_zw2.setStyleSheet('QPushButton {font-size:10px}')
        self.buttom_zw2.clicked.connect(lambda: self.button_click("-z world"))
        

        # World (r,p,y) ---------------------------------------------------------------------------------------------

        # button Roll world location
        self.buttom_rw1 = QPushButton("Roll++", self)
        self.buttom_rw1.move(140,30)
        self.buttom_rw1.resize(50,20)
        self.buttom_rw1.setStyleSheet('QPushButton {font-size:10px}')
        self.buttom_rw1.clicked.connect(lambda: self.button_click("roll world"))

        self.buttom_rw2 = QPushButton("Roll--", self)
        self.buttom_rw2.move(140,52)
        self.buttom_rw2.resize(50,20)
        self.buttom_rw2.setStyleSheet('QPushButton {font-size:10px}')
        self.buttom_rw2.clicked.connect(lambda: self.button_click("-roll world"))

        # button Pit world location
        self.buttom_pw1 = QPushButton("Pit++", self)
        self.buttom_pw1.move(140,80)
        self.buttom_pw1.resize(50,20)
        self.buttom_pw1.setStyleSheet('QPushButton {font-size:10px}')
        self.buttom_pw1.clicked.connect(lambda: self.button_click("pit world"))

        self.buttom_pw2 = QPushButton("Pit--", self)
        self.buttom_pw2.move(140,102)
        self.buttom_pw2.resize(50,20)
        self.buttom_pw2.setStyleSheet('QPushButton {font-size:10px}')
        self.buttom_pw2.clicked.connect(lambda: self.button_click("-pit world"))

        # button Yaw world location
        self.buttom_yaw1 = QPushButton("Yaw++", self)
        self.buttom_yaw1.move(140,130)
        self.buttom_yaw1.resize(50,20)
        self.buttom_yaw1.setStyleSheet('QPushButton {font-size:10px}')
        self.buttom_yaw1.clicked.connect(lambda: self.button_click("yaw world"))

        self.buttom_yaw1 = QPushButton("Yaw--", self)
        self.buttom_yaw1.move(140,152)
        self.buttom_yaw1.resize(50,20)
        self.buttom_yaw1.setStyleSheet('QPushButton {font-size:10px}')
        self.buttom_yaw1.clicked.connect(lambda: self.button_click("-yaw world"))


        # camera (x,y,z) ---------------------------------------------------------------------------------------------

        # button X camera location
        self.buttom_xc1 = QPushButton("X++", self)
        self.buttom_xc1.move(305,30)
        self.buttom_xc1.resize(25,20)
        self.buttom_xc1.setStyleSheet('QPushButton {font-size:10px}')
        self.buttom_xc1.clicked.connect(lambda: self.button_click("x camera"))

        self.buttom_xc2 = QPushButton("X--", self)
        self.buttom_xc2.move(305,52)
        self.buttom_xc2.resize(25,20)
        self.buttom_xc2.setStyleSheet('QPushButton {font-size:10px}')
        self.buttom_xc2.clicked.connect(lambda: self.button_click("-x camera"))

        # button Y camera location
        self.buttom_yc1 = QPushButton("Y++", self)
        self.buttom_yc1.move(305,80)
        self.buttom_yc1.resize(25,20)
        self.buttom_yc1.setStyleSheet('QPushButton {font-size:10px}')
        self.buttom_yc1.clicked.connect(lambda: self.button_click("y camera"))

        self.buttom_yc2 = QPushButton("Y--", self)
        self.buttom_yc2.move(305,102)
        self.buttom_yc2.resize(25,20)
        self.buttom_yc2.setStyleSheet('QPushButton {font-size:10px}')
        self.buttom_yc2.clicked.connect(lambda: self.button_click("-y camera"))

        # button Z camera location
        self.buttom_zc1 = QPushButton("Z++", self)
        self.buttom_zc1.move(305,130)
        self.buttom_zc1.resize(25,20)
        self.buttom_zc1.setStyleSheet('QPushButton {font-size:10px}')
        self.buttom_zc1.clicked.connect(lambda: self.button_click("z camera"))

        self.buttom_zc2 = QPushButton("Z--", self)
        self.buttom_zc2.move(305,152)
        self.buttom_zc2.resize(25,20)
        self.buttom_zc2.setStyleSheet('QPushButton {font-size:10px}')
        self.buttom_zc2.clicked.connect(lambda: self.button_click("-z camera"))


        # camera (r,p,y) ---------------------------------------------------------------------------------------------

        # button Roll camera location
        self.buttom_rc1 = QPushButton("Roll++", self)
        self.buttom_rc1.move(440,30)
        self.buttom_rc1.resize(50,20)
        self.buttom_rc1.setStyleSheet('QPushButton {font-size:10px}')
        self.buttom_rc1.clicked.connect(lambda: self.button_click("roll camera"))

        self.buttom_rc2 = QPushButton("Roll--", self)
        self.buttom_rc2.move(440,52)
        self.buttom_rc2.resize(50,20)
        self.buttom_rc2.setStyleSheet('QPushButton {font-size:10px}')
        self.buttom_rc2.clicked.connect(lambda: self.button_click("-roll camera"))

        # button Pit camera location
        self.buttom_pc1 = QPushButton("Pit++", self)
        self.buttom_pc1.move(440,80)
        self.buttom_pc1.resize(50,20)
        self.buttom_pc1.setStyleSheet('QPushButton {font-size:10px}')
        self.buttom_pc1.clicked.connect(lambda: self.button_click("pit camera"))

        self.buttom_pc2 = QPushButton("Pit--", self)
        self.buttom_pc2.move(440,102)
        self.buttom_pc2.resize(50,20)
        self.buttom_pc2.setStyleSheet('QPushButton {font-size:10px}')
        self.buttom_pc2.clicked.connect(lambda: self.button_click("-pit camera"))

        # button Yaw camera location
        self.buttom_yac1 = QPushButton("Yaw++", self)
        self.buttom_yac1.move(440,130)
        self.buttom_yac1.resize(50,20)
        self.buttom_yac1.setStyleSheet('QPushButton {font-size:10px}')
        self.buttom_yac1.clicked.connect(lambda: self.button_click("yaw camera"))

        self.buttom_yac2 = QPushButton("Yaw--", self)
        self.buttom_yac2.move(440,152)
        self.buttom_yac2.resize(50,20)
        self.buttom_yac2.setStyleSheet('QPushButton {font-size:10px}')
        self.buttom_yac2.clicked.connect(lambda: self.button_click("-yaw camera"))


        # Reset --------------------------------------------------------------------------------------

        # reset position 
        self.buttom_re1 = QPushButton("Reset position", self)
        self.buttom_re1.move(340,180)
        self.buttom_re1.resize(100,20)
        self.buttom_re1.setStyleSheet('QPushButton {font-size:10px}')
        self.buttom_re1.clicked.connect(lambda: self.button_click("reset position"))

        self.buttom_re2 = QPushButton("Reset parameter", self)
        self.buttom_re2.move(790,180)
        self.buttom_re2.resize(100,20)
        self.buttom_re2.setStyleSheet('QPushButton {font-size:10px}')
        self.buttom_re2.clicked.connect(lambda: self.button_click("reset parameter"))


    def button_click(self, action): 


        # button World -----------------------------------------------------------------------------------

        # button x world
        if action == "x world":
            x = self.text_xw.text()
            reference = "world"
            self.transformation(None,reference,float(x),0,0,0,0,0)

        # button -x world
        elif action == "-x world":
            x = self.text_xw.text()
            reference = "world"
            self.transformation(None,reference,-float(x),0,0,0,0,0)
        
        # button y world
        elif action == "y world":
            y = self.text_yw.text()
            reference = "world"
            self.transformation(None,reference,0,float(y),0,0,0,0)

        # button -y world
        elif action == "-y world":
            y = self.text_yw.text()
            reference = "world"
            self.transformation(None,reference,0,-float(y),0,0,0,0)

        # button z world
        elif action == "z world":
            z = self.text_zw.text() 
            reference = "world"
            self.transformation(None,reference,0,0,float(z),0,0,0)
        
        # button -z world
        elif action == "-z world":
            z = self.text_zw.text() 
            reference = "world"
            self.transformation(None,reference,0,0,-float(z),0,0,0)

        # button roll world
        elif action == "roll world":
            r = self.text_rw.text()
            axis = "x"  
            reference = "world"
            self.transformation(axis,reference,0,0,0,float(r),0,0)

        # button -roll world
        elif action == "-roll world":
            r = self.text_rw.text()
            axis = "x"  
            reference = "world"
            self.transformation(axis,reference,0,0,0,-float(r),0,0)

        # button pit world
        elif action == "pit world":
            p = self.text_pw.text()
            axis = "y" 
            reference = "world"
            self.transformation(axis,reference,0,0,0,0,float(p),0)

        # button -pit world
        elif action == "-pit world":
            p = self.text_pw.text()
            axis = "y" 
            reference = "world"
            self.transformation(axis,reference,0,0,0,0,-float(p),0)

        # button yaw world
        elif action == "yaw world":
            ya = self.text_yaw.text()
            axis = "z"
            reference = "world"
            self.transformation(axis,reference,0,0,0,0,0,float(ya))

        # button -yaw world
        elif action == "-yaw world":
            ya = self.text_yaw.text()
            axis = "z"
            reference = "world"
            self.transformation(axis,reference,0,0,0,0,0,-float(ya))


        # Buttom Camera --------------------------------------------------------------------------------------------------

        # button x camera
        elif action == "x camera":
            x = self.text_xc.text()
            reference = "camera"
            self.transformation(None,reference,float(x),0,0,0,0,0)

        # button -x camera
        elif action == "-x camera":
            x = self.text_xc.text()
            reference = "camera"
            self.transformation(None,reference,-float(x),0,0,0,0,0)

        # button y camera
        elif action == "y camera":
            y = self.text_yc.text()
            reference = "camera"
            self.transformation(None,reference,0,float(y),0,0,0,0)

        # button -y camera
        elif action == "-y camera":
            y = self.text_yc.text()
            reference = "camera"
            self.transformation(None,reference,0,-float(y),0,0,0,0)

        # button z camera
        elif action == "z camera":
            z = self.text_zc.text()
            reference = "camera"
            self.transformation(None,reference,0,0,float(z),0,0,0)

        # button -z camera
        elif action == "-z camera":
            z = self.text_zc.text()
            reference = "camera"
            self.transformation(None,reference,0,0,-float(z),0,0,0)

        # button roll camera
        elif action == "roll camera":
            r = self.text_rc.text()
            axis = "x"
            reference = "camera"
            self.transformation(axis,reference,0,0,0,float(r),0,0)

        # button -roll camera
        elif action == "-roll camera":
            r = self.text_rc.text()
            axis = "x"
            reference = "camera"
            self.transformation(axis,reference,0,0,0,-float(r),0,0)
        
        # button pit camera
        elif action == "pit camera":
            p = self.text_pc.text()
            axis = "y"
            reference = "camera"
            self.transformation(axis,reference,0,0,0,0,float(p),0)

        # button -pit camera
        elif action == "-pit camera":
            p = self.text_pc.text()
            axis = "y"
            reference = "camera"
            self.transformation(axis,reference,0,0,0,0,-float(p),0)
        
        # button yaw camera
        elif action == "yaw camera":
            ya = self.text_yac.text()
            axis = "z"
            reference = "camera"
            self.transformation(axis,reference,0,0,0,0,0,float(ya))

        # button -yaw camera
        elif action == "-yaw camera":
            ya = self.text_yac.text()
            axis = "z"
            reference = "camera"
            self.transformation(axis,reference,0,0,0,0,0,-float(ya))


        # Button Reset----------------------------------------------------------------------------------------

        # button reset camera  
        elif action == "reset position":
            self.object()
            self.plot3d()
            self.projection()
            self.plot2d()

        elif action == "reset parameter":
            self.camera_parameter()
            self.sp_f.setValue(self.f)
            self.sp_sx.setValue(self.sx)
            self.sp_sy.setValue(self.sy)
            self.sp_s0.setValue(self.s0)
            self.sp_ox.setValue(int(self.ox))
            self.sp_oy.setValue(int(self.oy))
            self.projection()


    def spinbox(self):

        # define f
        self.sp_f = QSpinBox(self)
        self.sp_f.setMinimum(1)
        self.sp_f.setMaximum(1000)
        self.sp_f.setValue(self.f)
        self.sp_f.setSingleStep(10)
        self.sp_f.move(605,40)
        self.sp_f.resize(50,25)
        self.sp_f.valueChanged.connect(lambda: self.spinbox_clicked("f"))

        # define sx
        self.sp_sx = QSpinBox(self)
        self.sp_sx.setMinimum(0)
        self.sp_sx.setMaximum(1000)
        self.sp_sx.setValue(self.sx)
        self.sp_sx.setSingleStep(10)
        self.sp_sx.move(605,90)
        self.sp_sx.resize(50,25)
        self.sp_sx.valueChanged.connect(lambda: self.spinbox_clicked("sx"))

        # define sy
        self.sp_sy = QSpinBox(self)
        self.sp_sy.setMinimum(0)
        self.sp_sy.setMaximum(1000)
        self.sp_sy.setValue(self.sy)
        self.sp_sy.setSingleStep(10)
        self.sp_sy.move(605,140)
        self.sp_sy.resize(50,25)
        self.sp_sy.valueChanged.connect(lambda: self.spinbox_clicked("sy"))

        # define s0
        self.sp_s0 = QSpinBox(self)
        self.sp_s0.setMinimum(-1000)
        self.sp_s0.setMaximum(1000)
        self.sp_s0.setValue(self.s0)
        self.sp_s0.setSingleStep(10)
        self.sp_s0.move(750,40)
        self.sp_s0.resize(50,25)
        self.sp_s0.valueChanged.connect(lambda: self.spinbox_clicked("s0"))

        # define ox
        self.sp_ox = QSpinBox(self)
        self.sp_ox.setMinimum(0)
        self.sp_ox.setMaximum(500)
        self.sp_ox.setValue(int(self.ox))
        self.sp_ox.setSingleStep(10)
        self.sp_ox.move(750,90)
        self.sp_ox.resize(50,25)
        self.sp_ox.valueChanged.connect(lambda: self.spinbox_clicked("ox"))

        # define oy
        self.sp_oy = QSpinBox(self)
        self.sp_oy.setMinimum(0)
        self.sp_oy.setMaximum(500)
        self.sp_oy.setValue(int(self.oy))
        self.sp_oy.setSingleStep(10)
        self.sp_oy.move(750,140)
        self.sp_oy.resize(50,25)
        self.sp_oy.valueChanged.connect(lambda: self.spinbox_clicked("oy"))


    def spinbox_clicked(self, action):

        if action == "f":
            self.f = self.sp_f.value()

        elif action == "sx":
            self.sx = self.sp_sx.value()

        elif action == "sy":
            self.sy = self.sp_sy.value()

        elif action == "s0":
            self.s0 = self.sp_s0.value()

        elif action == "ox":
            self.ox = self.sp_ox.value()

        elif action == "oy":
            self.oy = self.sp_oy.value()

        self.projection()


    def loading_window(self):

        self.setGeometry(self.w, self.a, self.largura, self.altura)
        self.setWindowTitle(self.title)
        self.show()


if __name__=="__main__":

    aplication = QApplication(sys.argv)
    window = App()
    sys.exit(aplication.exec_())