import pinocchio as pin
import numpy as np
import example_robot_data as robex
import matplotlib.pyplot as plt
from numpy.linalg import norm,inv,pinv,svd,eig
import talos_low
import tkinter as tk
import time
from cop2forces import ContactForceDistributor

GUESS_FILE = '/tmp/sol.npy'
DT = 0.01
robot = talos_low.load()
model = robot.model
data = model.createData()
model.q0 = robot.q0

contactIds = [ i for i,f in enumerate(model.frames) if "sole_link" in f.name ]

viz = pin.visualize.GepettoVisualizer(robot.model,robot.collision_model,robot.visual_model)
viz.initViewer()
viz.loadViewerModel()
gv = viz.viewer.gui

sol = np.load(GUESS_FILE,allow_pickle=True)[()]
xs = sol['xs']
fs = sol['fs']
taus = sol['us']
acs = sol['acs']

#

size = 0.205,0.122
footShape = np.array( [
    [ size[0]/2, size[1]/2,0],
    [-size[0]/2, size[1]/2,0],
    [-size[0]/2,-size[1]/2,0],
    [ size[0]/2,-size[1]/2,0] ])
forceScaler = 1./500

# -------------------------------------------------------------------------------------
'''
def disp(t,withForce=True,scale=None):
    t=int(t)
    assert(t>=0 and t<len(xs))
    viz.display( xs[t,:model.nq] )

    if withForce and t<len(fs):
        phis=np.split(fs[t],len(contactIds))
        pin.framesForwardKinematics(model,data,xs[t][:model.nq])

        for cid,phi in zip(contactIds,phis):
            assert(len(phi)==6)
            fk = contactForceDistribution(phi,footShape)
            oMf = data.oMf[cid]
            for k,(f,p) in enumerate(zip(fk,footShape)):
                name = f'world/line{model.frames[cid].name}_{k}'
                p0 = oMf*p
                p1 = p0 + oMf.rotation@f*forceScaler
                try:
                    gv.setLineStartPoint(name,p0.tolist())
                    gv.setLineEndPoint(name,p1.tolist())
                except:
                    gv.addLine(name,p0.tolist(),p1.tolist(),[1,0,0,1])

    if scale is not None:
        scale.set(t)
'''
class Display:
    def __init__(self):
        self.contactForceDistribution = ContactForceDistributor(footShape)
        self.dispForceFreq = 1
        self.withForces=True
        self.disp(0)
    def __call__(self,t,scale=None):
        return self.disp(t,scale)
    def disp(self,t,scale=None):
        t=int(t)
        assert(t>=0 and t<len(xs))
        viz.display( xs[t,:model.nq] )
        if self.withForces and t<len(fs):
            if not t % self.dispForceFreq:
                self.dispForce(t)
        if scale is not None:
            scale.set(t)
    def getForceName(self,contactId,forceId):
        return  f'world/line{model.frames[contactId].name}_{forceId}'
    def changeWithForce(self):
        print('CHECKED',self.withForces)
        self.withForces = not self.withForces
        for cid in contactIds:
            for k in range(len(footShape)):
                gv.setVisibility(self.getForceName(cid,k),'ON' if self.withForces else 'OFF')
        if self.withForces:
            t=int(timeScale.get())
            if t<len(fs):
                self.dispForce(t)
    def dispForce(self,t):
        phis=np.split(fs[t],len(contactIds))
        pin.framesForwardKinematics(model,data,xs[t][:model.nq])

        for cid,phi in zip(contactIds,phis):
            assert(len(phi)==6)
            fk = self.contactForceDistribution.solve(phi)
            oMf = data.oMf[cid]
            for k,(f,p) in enumerate(zip(fk,footShape)):
                name = self.getForceName(cid,k)
                p0 = oMf*p
                p1 = p0 + oMf.rotation@f*forceScaler
                try:
                    gv.setLineStartPoint(name,p0.tolist())
                    gv.setLineEndPoint(name,p1.tolist())
                    pass
                except:
                    gv.addLine(name,p0.tolist(),p1.tolist(),[1,0,0,1])
disp = Display()


robot = talos_low.load()
class Player():
    def __init__(self,timeScale):
        self.playing = False
        self.timeScale = timeScale
    def play(self):
        self.playing = True
        for t,x in enumerate(xs):
            if t<self.timeScale.get(): continue
            if not self.playing:
                break
            disp(t,scale=self.timeScale)
            root.update()
            time.sleep(DT)
        self.playing = False
    def stop(self):
        self.playing=False
        self.timeScale.set(0)
    def pause(self):
        self.playing=False
    def next(self):
        if self.playing: return
        t = int(self.timeScale.get())
        if t+1>=len(xs): return
        disp(t+1,scale=self.timeScale)
    def prev(self):
        if self.playing: return
        t = int(self.timeScale.get())
        if t<=0: return
        disp(t-1,scale=self.timeScale)
    def playpause(self):
        if self.playing: self.pause()
        else: self.play()

root = tk.Tk()
timeScale = tk.Scale(root, from_=0, to=len(xs)-1,command=disp,
             orient=tk.HORIZONTAL,label='time',resolution=1,tickinterval=10,length=500)
timeScale.pack()
frame = tk.Frame(root)
frame.pack()
play = Player(timeScale)
playButton = tk.Button(frame,text='play',command=play.play)
playButton .grid(row=0,column=0)
stopButton = tk.Button(frame,text='stop',command=play.stop)
stopButton .grid(row=0,column=1)
pauseButton = tk.Button(frame,text='pause',command=play.pause)
pauseButton .grid(row=0,column=2)

checkFrame = tk.Frame(root)
checkFrame.pack()
checkForce = tk.Checkbutton(checkFrame, text='show forces',
                            command=disp.changeWithForce)
checkForce.grid(column=0,row=0)
checkForce.select()

VIEW_FILE =  'tkviews.npy'
views = {}
try:
    views = np.load(VIEW_FILE,allow_pickle=True)[()]
except:
    print('Error while loading tkviews.npy')

class ChangeView:
    def __init__(self,viewName):
        self.viewName = viewName
    def __call__(self):
        print(f'Change for {self.viewName}')
        gv.setCameraTransform(viz.windowID,views[self.viewName])
def saveView():
    print(f'Add view {entryName.get()}')
    views[entryName.get()] = gv.getCameraTransform(viz.windowID)
    np.save(open(VIEW_FILE, "wb"),views)
    
viewFrame = tk.Frame(root)
viewFrame.pack()
for n in views.keys():
    view1 = tk.Button(viewFrame,text=n,command=ChangeView(n))
    view1.pack(side=tk.LEFT)#
    #view1.grid(column=0,row=0)

confFrame = tk.Frame(root)
confFrame.pack()
tk.Label(confFrame,text='Add a view: ').grid(column=0,row=0)
entryName = tk.Entry(confFrame,text='Name:')
entryName.grid(column=1,row=0)
#tk.Label(confFrame,text='id').grid(column=2,row=0)
#entryNumber = tk.Entry(confFrame,text='Id:')
#entryNumber.grid(column=3,row=0)
tk.Button(confFrame,text='Save',command=saveView).grid(column=4,row=0)

root.bind('<Escape>',lambda event: root.destroy())
root.bind('<Right>',lambda event: play.next())
root.bind('<Left>',lambda event: play.prev())
root.bind('<space>',lambda event: play.playpause())
root.bind('<Home>',lambda event: play.stop())

tk.mainloop()


