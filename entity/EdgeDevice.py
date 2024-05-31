class Device:

    def __init__(self, deviceId):
        self.deviceId = deviceId


class AccessPoint(Device):

    def __init__(self, deviceId, arrivalRate, wirelessTransRate):
        super(AccessPoint, self).__init__(deviceId)
        self.deviceId = deviceId
        self.arrivalRate = arrivalRate
        self.wirelessTransRate = wirelessTransRate


class Server(Device):

    def __init__(self, deviceId, capacity):
        super(Server, self).__init__(deviceId)
        self.deviceId = deviceId
        self.capacity = capacity


class Switch(Device):

    def __init__(self, deviceId):
        super(Switch, self).__init__(deviceId)
        self.deviceId = deviceId