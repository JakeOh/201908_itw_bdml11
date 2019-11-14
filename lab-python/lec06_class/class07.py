class Account:
    """ 은행 계좌 클래스
    field(데이터): 계좌번호(accountno), 잔액(balance)
    method(기능): 입금(deposit), 출금(withdraw), 이체(transfer)
    """
    def __init__(self, accountno, balance):
        self.accountno = accountno
        self.balance = balance

    def __repr__(self):
        return f'Account(account no.: {self.accountno}, balance: {self.balance})'

    def deposit(self, money):
        # if money < 0:
        #     raise ValueError('입금 금액은 0보다 크거나 같아야...')
        self.balance += money
        print(f'{money} 입금 후 잔액: {self.balance}')

    def withdraw(self, money):
        # if money < 0:
        #     raise ...
        # elif self.balance < money:
        #     raise
        self.balance -= money
        print(f'{money} 출금 후 잔액: {self.balance}')

    def transfer(self, to, money):
        """
        계좌 이체 기능. 내 계좌에서 money를 출금해서 상대방(to) 계좌에 입금.

        :param to: 이체할 상대방 계좌(Account 클래스 객체)
        :param money: 이체할 금액(숫자)
        :return: None
        """
        self.withdraw(money)  # 내 계좌에서 출금
        to.deposit(money)  # 상대방 계좌에 입금


if __name__ == '__main__':
    account1 = Account(123456, 1000)
    print(account1)
    account1.deposit(500)
    account1.withdraw(100)

    account2 = Account(789000, 100)
    account1.transfer(account2, 500)

    print(account1)
    print(account2)
