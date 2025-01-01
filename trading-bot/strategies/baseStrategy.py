
class Strategy:
    def __init__(self, entryPrice, exitPrice, entryDate, exitDate, currPrice, currDate, id):
        self.id = id
        self.entryPrice = entryPrice
        self.exitPrice = exitPrice
        self.entryDate = entryDate
        self.exitDate = exitDate
        self.currPrice = currPrice
        self.currDate = currDate
    def display(self):
        print("Stock Price Entry: " + self.entryPrice)
        print("Current Stock Price:" + self.currPrice)


        