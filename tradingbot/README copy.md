# TODO

2025-03-20

- Visualize custom hammer chart on web app *DONE*
- Visualize support and resistance on web app *DONE*
- Visualize buy positions on web app
    - We should create a strategy class first that can be used interchangebly between the web app and trading bot

- Review custom hammer on bot *DONE*

- Continue to test strategies for today *DONE* 

- Move all work over to backtesting folder *DONE*

- Initialize trading with the alpaca trade API

DEBUGGING

The values being passed into the new hammer function for the trading bot are <class 'numpy.ndarray'>



2025-03-21

- Create new list of stocks *DONE*
- Create a strategy that uses supports and resistances
- A strategy that further refines uses of the trend line
- Do some research on further understanding swing trading

We learned something today, you need to know what kind of trading you are looking for.
Swing trading or Day trading, or whatever else kind of trading. All or most parameters 
must be changed. 

2025-03-23

- Visualize buy positions on web app
    - We should create a strategy class first that can be used interchangebly between the web app and trading bot *DONE*

2025-03-25

- Analyze the current test to identify the multiple losses. I was getting burned from the dips. 
- A strategy that further refines uses of the trend line

2025-03-26

We may have to give up on the charting aspect, the trading bot has gotten too big to integrate with the chart. May need to start a new chart with a fully developed trading bot. Pain

2025-04-03

Completed the backtesting. Anymore and we would just be approaching scope creep. The charts have been created. 

It's best to use Alpaca websockets api to fetch data and send buy signals


- Create a live websockets script to send data to the bot. *DONE*
- Redo the web app to display the live data.






