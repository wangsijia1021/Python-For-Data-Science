import json
import os
import math
import numpy as np
# import matplotlib.pyplot as plt
# import pylab as pyl
import scipy.io as sio
# import datetime


class resolveDataCsv(object):
    def __init__(self, Jpath, savePath=''):
        self.savePath = savePath
        self.count = 0
        self.Jpath = Jpath
        self.loadFromCsv(self.Jpath)


    def loadFromCsv(self,Jpath):

        alttitudeFileNameY = r'motion_attitude_all0.csv'
        alttitudeFileNameW = r'motion_attitude_all1.csv'
        alttitudeFileNameZ = r'motion_attitude_all2.csv'
        alttitudeFileNameX = r'motion_attitude_all3.csv'
        rotationFileNameX = r'motion_rotation_all0.csv'
        rotationFileNameY = r'motion_rotation_all1.csv'
        rotationFileNameZ = r'motion_rotation_all2.csv'
        accelerationFileNameX = r'motion_useracceleration_x.csv'
        accelerationFileNameY = r'motion_useracceleration_y.csv'
        accelerationFileNameZ = r'motion_useracceleration_z.csv'
        infoFileName = r'motion_info_all.csv'
        # SourceAlX = open(os.path.join(Jpath,alttitudeFileNameX)).read().split('\n')
        # SourceAlY = open(os.path.join(Jpath, alttitudeFileNameY)).read().split('\n')
        # SourceAlZ = open(os.path.join(Jpath, alttitudeFileNameZ)).read().split('\n')
        # SourceAlW = open(os.path.join(Jpath, alttitudeFileNameW)).read().split('\n')
        # SourceAccX =open(os.path.join(Jpath,accelerationFileNameX)).read().split('\n')
        # SourceAccY = open(os.path.join(Jpath, accelerationFileNameY)).read().split('\n')
        # SourceAccZ = open(os.path.join(Jpath, accelerationFileNameZ)).read().split('\n')
        # SourceRRX = open(os.path.join(Jpath, rotationFileNameX)).read().split('\n')

        SourceAlX = open(os.path.join(Jpath,alttitudeFileNameX))
        SourceAlY = open(os.path.join(Jpath, alttitudeFileNameY))
        SourceAlZ = open(os.path.join(Jpath, alttitudeFileNameZ))
        SourceAlW = open(os.path.join(Jpath, alttitudeFileNameW))
        SourceAccX =open(os.path.join(Jpath,accelerationFileNameX))
        SourceAccY = open(os.path.join(Jpath, accelerationFileNameY))
        SourceAccZ = open(os.path.join(Jpath, accelerationFileNameZ))
        SourceRRX = open(os.path.join(Jpath, rotationFileNameX))
        SourceRRY = open(os.path.join(Jpath, rotationFileNameY))
        SourceRRZ = open(os.path.join(Jpath, rotationFileNameZ))
        dataInfoArr = open(os.path.join(Jpath,infoFileName)).read().split('\n')

        # drop the item header
        SourceAlX.readline()
        SourceAlY.readline()
        SourceAlZ.readline()
        SourceAlW.readline()
        SourceAccX.readline()
        SourceAccY.readline()
        SourceAccZ.readline()
        SourceRRX.readline()
        SourceRRY.readline()
        SourceRRZ.readline()

        # data = []
        count  = 0

        for dataInfoIterator in dataInfoArr:
            datai = []
            data = []
            if count == 0:
                count+=1
                continue
            dataInfo = dataInfoIterator.split(',')
            healthCode = dataInfo[0]
            # lengthInfo = dataInfo[1]
            recordID = dataInfo[2]
            if len(healthCode) ==0:
                continue

            #sample data
            #{"attitude":{"y":-0.03907553041154967,"w":0.9989387013592812,"z":2.624416970342931E-5,"x":-0.02438387091772637},"timestamp":70502.14567362501,"rotationRate":{"x":0.1088925823569298,"y":0.02010470628738403,"z":0.006740015931427479},"userAcceleration":{"x":0.01303387992084026,"y":0.004123150836676359,"z":-0.00355629506520927}




            alx = SourceAlX.readline().split(',')
            aly = SourceAlY.readline().split(',')
            alz = SourceAlZ.readline().split(',')
            alw = SourceAlW.readline().split(',')
            accx = SourceAccX.readline().split(',')
            accy = SourceAccY.readline().split(',')
            accz = SourceAccZ.readline().split(',')
            rrx = SourceRRX.readline().split(',')
            rry = SourceRRY.readline().split(',')
            rrz = SourceRRZ.readline().split(',')
            datalen =min(len(alw), len(alx), len(aly), len(alz), len(accx), len(accy), len(accz), len(rrx), len(rry), len(rrz))
            for j in range(datalen):
                Jal = {"y": 0, "w": 0, "z": 0, "x": 0}
                Jacc = {"x": 0, "y": 0, "z": 0}
                Jrr = {"x": 0, "y": 0, "z": 0}
                dataj = {"attitude": Jal, "rotationRate": Jrr, "userAcceleration": Jacc, "timestamp": 0}
                timestamp = 0.0
                dataj["attitude"]["x"] = float(alx[j])
                dataj["attitude"]["y"] = float(aly[j])
                dataj["attitude"]["z"] = float(alz[j])
                dataj["attitude"]["w"] = float(alw[j])
                dataj["rotationRate"]["x"] = float(rrx[j])
                dataj["rotationRate"]["y"] = float(rry[j])
                dataj["rotationRate"]["z"] = float(rrz[j])
                dataj["userAcceleration"]["x"] = float(accx[j])
                dataj["userAcceleration"]["y"] = float(accy[j])
                dataj["userAcceleration"]["z"] = float(accz[j])
                dataj["timestamp"] = float(timestamp)
                timestamp+=1.0/90.0
                datai.append(dataj)
            data.append(datai)
            # self.patten = 7
            tempAcc, tempRR = self.resolve(data)
            sio.savemat(os.path.join(str(self.savePath), str(healthCode) + '_' + str(recordID) + '_' + 'Acc' + '.mat'),
                        {'A': tempAcc})
            sio.savemat(os.path.join(str(self.savePath), str(healthCode) + '_' + str(recordID) +  '_' + 'RR' + '.mat'),
                        {'A': tempRR})
            print('%s is done'%healthCode)

        SourceAlX.close()
        SourceAlY.close()
        SourceAlZ.close()
        SourceAlW.close()
        SourceAccX.close()
        SourceAccY.close()
        SourceAccZ.close()
        SourceRRX.close()
        SourceRRY.close()
        SourceRRZ.close()


    ##########################functions used in resolve
    def quaternionMultiply(self, quaternion1, quaternion2):
        quaternionResult = [0, 0, 0, 0]
        quaternionResult[0] = quaternion1[3] * quaternion2[0] + quaternion1[0] * quaternion2[3] + quaternion1[1] * \
                                                                                                  quaternion2[2] - \
                              quaternion1[2] * quaternion2[1]
        quaternionResult[1] = quaternion1[3] * quaternion2[1] - quaternion1[0] * quaternion2[2] + quaternion1[1] * \
                                                                                                  quaternion2[3] + \
                              quaternion1[2] * quaternion2[0]
        quaternionResult[2] = quaternion1[3] * quaternion2[2] + quaternion1[0] * quaternion2[1] - quaternion1[1] * \
                                                                                                  quaternion2[0] + \
                              quaternion1[2] * quaternion2[3]
        quaternionResult[3] = quaternion1[3] * quaternion2[3] - quaternion1[0] * quaternion2[0] - quaternion1[1] * \
                                                                                                  quaternion2[1] - \
                              quaternion1[2] * quaternion2[2]
        return quaternionResult

    def rotation3D(self, vector, xr, yr, zr):
        xVector = vector.dot(np.linalg.inv(xr))
        yVector = xVector.dot(np.linalg.inv(yr))
        zVector = yVector.dot(np.linalg.inv(zr))
        return zVector

    def calculateIncludedAngle(self, vector1, vector2):
        Mol1 = math.sqrt(vector1[0] ** 2 + vector1[1] ** 2 + vector1[2] ** 2)
        Mol2 = math.sqrt(vector2[0] ** 2 + vector2[1] ** 2 + vector2[2] ** 2)
        newVector1 = vector1
        newVector2 = vector2
        cosAngle = newVector1.dot(np.transpose(newVector2)) / (Mol1 * Mol2)
        # print('V1:%s V2:%s\ncosAngle:%f A.dotB:%f Mol(A):%f Mol(B):%f'%(str(newVector1),str(newVector2),cosAngle,vector1.dot(vector2),Mol1,Mol2))
        # print('V1:%f %f %f     V2:%f %f %f      cosAngle: %f'%(newVector1[0],newVector1[1],newVector1[2],newVector2[0],
        #                                                         newVector2[1],newVector2[2],cosAngle))
        if -1 < cosAngle < 1:
            Angle = math.acos(cosAngle) * math.pi / 180.0
        elif abs(cosAngle - 1) < 1e-3:
            Angle = math.pi
        elif cosAngle == -1:
            Angle = -math.pi
        return Angle

    def calculateR(self, q):
        s = 1 / (q[0] ** 2 + q[1] ** 2 + q[3] ** 2 + q[2] ** 2)
        qr = q[3]
        qi = q[0]
        qj = q[1]
        qk = q[2]
        r = np.array([[1 - 2 * s * (qj ** 2 + qk ** 2), 2 * s * (qi * qj - qk * qr), 2 * s * (qi * qk + qj * qr)],
                      [2 * s * (qi * qj + qk * qr), 1 - 2 * s * (qi ** 2 + qk ** 2), 2 * s * (qj * qk - qi * qr)],
                      [2 * s * (qi * qk - qj * qr), 2 * s * (qj * qk + qi * qr), 1 - 2 * s * (qi ** 2 + qj ** 2)]],
                     dtype='float64')
        return r

    def componentX(self, Vector):
        return np.array([0, Vector[1], Vector[2]], dtype='float64')

    def componentY(self, Vector):
        return np.array([Vector[0], 0, Vector[2]], dtype='float64')

    def componentZ(self, Vector):
        return np.array([Vector[0], Vector[1], 0], dtype='float64')

        ######################main resolve function

    def resolve(self, data):
        # if data == None:
        #     data = self.data
        fileNum = len(data)
        phoneRotationOfAll = []

        # for i in range(fileNum):
        for i in range(1):
            personNum = len(data[i])
            one = 1
            fone = float(one)
            q = [fone, fone, fone, fone]
            phoneRotationOfOne = []
            timeStamp = []

            measureRotationRate = np.empty((personNum, 3, 1), dtype='float64')
            measureAcceleration = np.empty((personNum, 3, 1), dtype='float64')
            realAcceleration = np.empty((personNum, 3), dtype='float64')
            realRotationRate = np.empty((personNum, 3), dtype='float64')
            realRotationRate2 = np.empty((personNum, 3), dtype='float64')
            realVelocity = np.empty((personNum, 1, 3), dtype='float64')
            frequency = np.empty((personNum), dtype='float64')
            gravity = np.empty((personNum, 3), dtype='float64')
            print('\n')
            for j in range(personNum):
                # if j == 0:
                #     timeStamp.append(data[i][j]['timestamp'])
                #     timeInterval = data[i][j + 1]['timestamp'] - timeStamp[0]
                # else:
                #     timeStamp.append(data[i][j]['timestamp'])
                #     timeInterval = timeStamp[j] - timeStamp[j - 1]

                timeInterval = 1.0/90.0
                    # frequency[j] = 1 / timeInterval
                #################calculate the quaternion of now
                #################emergency warning of math(you can skip this)
                # this is wrong look at the next
                # q[0] = q[0] * data[i][j]['attitude']['x']
                # q[1] = q[1] * data[i][j]['attitude']['y']
                # q[2] = q[2] * data[i][j]['attitude']['z']
                # q[3] = q[3] * data[i][j]['attitude']['w']

                # this is right (hu finally work)yeah!
                qNew = [0, 0, 0, 0]
                qNew[0] = data[i][j]['attitude']['x']
                qNew[1] = data[i][j]['attitude']['y']
                qNew[2] = data[i][j]['attitude']['z']
                qNew[3] = data[i][j]['attitude']['w']
                q= qNew
                if ((q[0] ** 2 + q[1] ** 2 + q[3] ** 2 + q[2] ** 2))==0:
                    continue
                r = self.calculateR(qNew)

                #################load the measure acc and rotationrate

                measureRotationRate[j, 0, 0] = data[i][j]['rotationRate']['x']
                measureRotationRate[j, 1, 0] = data[i][j]['rotationRate']['y']
                measureRotationRate[j, 2, 0] = data[i][j]['rotationRate']['z']

                measureAcceleration[j, 0, 0] = data[i][j]['userAcceleration']['x']
                measureAcceleration[j, 1, 0] = data[i][j]['userAcceleration']['y']
                measureAcceleration[j, 2, 0] = data[i][j]['userAcceleration']['z']
                # measureAcceleration[j,3] = 1.0

                # oldGravity = np.array([0, 0, 0], dtype='float64')
                # oldGravity[0] = data[i][j]['gravity']['x']
                # oldGravity[1] = data[i][j]['gravity']['y']
                # oldGravity[2] = data[i][j]['gravity']['z']

                realRotationRate[j, :] = np.transpose(r.dot(measureRotationRate[j, :, 0]))
                realAcceleration[j, :] = np.transpose(r.dot(measureAcceleration[j, :, 0]))
                # gravity[j, :] = np.transpose(r.dot(np.transpose(oldGravity)))

                ####################
                # print('timeInter:%f     angle:%f \n'%(timeInterval,(180/math.pi)*measureRotationRate[j,0,0]*timeInterval))
                xVector = np.array([0, math.cos((180 / math.pi) * measureRotationRate[j, 0, 0] * timeInterval),
                                    math.sin((180 / math.pi) * measureRotationRate[j, 0, 0] * timeInterval)],
                                   dtype='float64')

                yVector = np.array([math.sin((180 / math.pi) * measureRotationRate[j, 1, 0] * timeInterval), 0,
                                    math.cos((180 / math.pi) * measureRotationRate[j, 1, 0] * timeInterval)],
                                   dtype='float64')

                zVector = np.array([math.cos((180 / math.pi) * measureRotationRate[j, 2, 0] * timeInterval),
                                    math.sin((180 / math.pi) * measureRotationRate[j, 2, 0] * timeInterval), 0],
                                   dtype='float64')

                # print('xV:%f %f %f   yV:%f %f %f  zV:%f %f %f\n'%(xVector[0], xVector[1], xVector[2],
                #                                                yVector[0], yVector[1], yVector[2],
                #                                                zVector[0], zVector[1], zVector[2],))

                newVectorX = np.transpose(r.dot(np.transpose(xVector)))  # fromY
                newVectorY = np.transpose(r.dot(np.transpose(yVector)))  # fromZ
                newVectorZ = np.transpose(r.dot(np.transpose(zVector)))  # fromX

                # print('newXV:%f %f %f   newYV:%f %f %f newZV:%f %f %f\n\n'%(newVectorX[0], newVectorX[1], newVectorX[2],
                #                                                newVectorY[0], newVectorY[1], newVectorY[2],
                #                                                newVectorZ[0], newVectorZ[1], newVectorZ[2],))
                oldAxisX = np.array([1, 0, 0], dtype='float64')
                oldAxisY = np.array([0, 1, 0], dtype='float64')
                oldAxisZ = np.array([0, 0, 1], dtype='float64')
                newAxisX = np.transpose(r.dot(np.transpose(oldAxisX)))
                newAxisY = np.transpose(r.dot(np.transpose(oldAxisY)))
                newAxisZ = np.transpose(r.dot(np.transpose(oldAxisZ)))

                AngleVXAY = np.empty((3), dtype='float64')
                AngleVYAZ = np.empty((3), dtype='float64')
                AngleVZAX = np.empty((3), dtype='float64')
                AngleVXAY[0] = self.calculateIncludedAngle(self.componentX(newVectorX), self.componentX(newAxisY)) * \
                               (1 if self.calculateIncludedAngle(self.componentX(newAxisY), oldAxisY) <
                                     self.calculateIncludedAngle(self.componentX(newVectorX), oldAxisY)else -1)
                AngleVXAY[1] = self.calculateIncludedAngle(self.componentY(newVectorX), self.componentY(newAxisY)) * \
                               (1 if self.calculateIncludedAngle(self.componentY(newAxisY), oldAxisZ) <
                                     self.calculateIncludedAngle(self.componentY(newVectorX), oldAxisZ)else -1)
                AngleVXAY[2] = self.calculateIncludedAngle(self.componentZ(newVectorX), self.componentZ(newAxisY)) * \
                               (1 if self.calculateIncludedAngle(self.componentZ(newAxisY), oldAxisX) <
                                     self.calculateIncludedAngle(self.componentZ(newVectorX), oldAxisX)else -1)

                AngleVYAZ[0] = self.calculateIncludedAngle(self.componentX(newVectorY), self.componentX(newAxisZ)) * \
                               (1 if self.calculateIncludedAngle(self.componentX(newAxisZ), oldAxisY) <
                                     self.calculateIncludedAngle(self.componentX(newVectorY), oldAxisY)else -1)
                AngleVYAZ[1] = self.calculateIncludedAngle(self.componentY(newVectorY), self.componentY(newAxisZ)) * \
                               (1 if self.calculateIncludedAngle(self.componentY(newAxisZ), oldAxisZ) <
                                     self.calculateIncludedAngle(self.componentY(newVectorY), oldAxisZ)else -1)
                AngleVYAZ[2] = self.calculateIncludedAngle(self.componentZ(newVectorY), self.componentZ(newAxisZ)) * \
                               (1 if self.calculateIncludedAngle(self.componentZ(newAxisZ), oldAxisX) <
                                     self.calculateIncludedAngle(self.componentZ(newVectorY), oldAxisX)else -1)

                AngleVZAX[0] = self.calculateIncludedAngle(self.componentX(newVectorZ), self.componentX(newAxisX)) * \
                               (1 if self.calculateIncludedAngle(self.componentX(newAxisX), oldAxisY) <
                                     self.calculateIncludedAngle(self.componentX(newVectorZ), oldAxisY)else -1)
                AngleVZAX[1] = self.calculateIncludedAngle(self.componentY(newVectorZ), self.componentY(newAxisX)) * \
                               (1 if self.calculateIncludedAngle(self.componentY(newAxisX), oldAxisZ) <
                                     self.calculateIncludedAngle(self.componentY(newVectorZ), oldAxisZ)else -1)
                AngleVZAX[2] = self.calculateIncludedAngle(self.componentZ(newVectorZ), self.componentZ(newAxisX)) * \
                               (1 if self.calculateIncludedAngle(self.componentZ(newAxisX), oldAxisX) <
                                     self.calculateIncludedAngle(self.componentZ(newVectorZ), oldAxisX)else -1)

                realRotationRate2[j, 0] = (AngleVXAY[0] + AngleVYAZ[0] + AngleVZAX[0]) / timeInterval
                realRotationRate2[j, 1] = (AngleVXAY[1] + AngleVYAZ[1] + AngleVZAX[1]) / timeInterval
                realRotationRate2[j, 2] = (AngleVXAY[2] + AngleVYAZ[2] + AngleVZAX[2]) / timeInterval

            return realAcceleration, realRotationRate


if __name__ == '__main__':

# change this to your PD motion Path
    Jpath = r'D:\parkinsonFullData\PD motion'
#change this to a temp path (keep the readPath in matlab file the same)
    savePath = os.path.join(r'C:\Users\Administrator\Desktop\Parkingson','tempData')

    rsd = resolveDataCsv(Jpath=Jpath, savePath=savePath)
