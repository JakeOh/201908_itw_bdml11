"""
클래스(class):
프로그램에서 만들려고 하는 대상(객체)이 가져야 할
속성(데이터)과 기능(함수)을 묶은 "데이터 타입"

메소드(method): 클래스가 가지고 있는 함수
필드(field): 클래스가 가지고 있는 데이터(변수)
"""

# TV 소프트웨어 작성
# TV 속성(데이터): 채널, 음량, 전원
# TV 기능: 채널 변경, 음량 변경, 전원 on/off
class BasicTv:
    """
    BasicTv 클래스
    """
    # 클래스 내부에서 선언하는 변수: field
    max_channel, min_channel = 5, 0
    max_volume, min_volume = 5, 0

    # 생성자가 호출됐을 때 실행되는 메소드(함수)
    def __init__(self, power, channel, volume):
        print('BasicTv 생성자 호출')
        self.power = power
        self.channel = channel
        self.volume = volume

    # 클래스 내부에서 정의하는 함수: 메소드
    def powerOnOff(self):
        if self.power:  # power가 True이면(TV가 켜져 있으면)
            self.power = False  # TV를 끔
            print('TV Off')
        else:  # TV 꺼져 있으면
            self.power = True  # TV를 켬
            print('TV On')

    def channelUp(self):
        # TV가 켜져 있는 경우에만 채널 변경(+1)
        if self.power:
            if self.channel < self.max_channel:
                # 현재 채널 값이 채널의 최댓값보다 작으면 +1
                self.channel += 1
            else:
                # 현재 채널 값이 채널 최댓값과 같으면 0으로 순환
                self.channel = self.min_channel
            print('Channel:', self.channel)

    def channelDown(self):
        # TV가 켜져 있는 상태에서만 채널 값을 -1
        if self.power:
            if self.channel > self.min_channel:
                # 현재 채널이 채널 최솟값보다 큰 경우에만 -1
                self.channel -= 1
            else:
                # 현재 채널이 최솟값인 경우는 채널 최댓값으로 순환
                self.channel = self.max_channel
            print('Channel:', self.channel)

    def volumeUp(self):
        # TV가 켜져 있는 경우에만 음량 +1
        if self.power:
            if self.volume < self.max_volume:
                # 현재 음량이 음량 최댓값보다 작은 경우 +1
                self.volume += 1
            # 음량 최댓값이면 아무 일 하지 않음(음량 유지)
            print('Volume:', self.volume)

    def volumeDown(self):
        # TV가 켜져 있을 때만 음량 -1
        if self.power:
            if self.volume > self.min_volume:
                # 현재 음량이 최솟값보다 큰 경우에만 -1
                self.volume -= 1
            # 현재 음량이 최솟값인 경우는 아무 일 하지 않음(유지)
            print('Volume:', self.volume)
# 클래스 설계(정의)


if __name__ == '__main__':
    # 클래스의 객체(인스턴스)를 생성해서 변수에 저장
    # 생성자(constructor) 호출 -> 객체(object) 생성
    tv1 = BasicTv(False, 0, 0)
    print('전원상태:', tv1.power)
    tv1.powerOnOff()  # TV를 켬
    tv1.channelUp()
    tv1.channelUp()
    tv1.channelUp()
    tv1.channelUp()
    tv1.channelUp()
    tv1.channelUp()
    for _ in range(10):
        tv1.volumeUp()
    for _ in range(10):
        tv1.volumeDown()




