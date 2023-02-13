import gradio as gr
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


import datetime
def get_time():
    return datetime.datetime.now().time()


counter = 0
start_year, test_year = 2018, 2023
datetime_column = "Date"
df_data = pd.read_csv(f"./data/EURUSD_Candlestick_1_M_BID_01.01.{start_year}-04.02.2023_processed.csv")
df_data[datetime_column] = pd.to_datetime(df_data[datetime_column], format="%Y-%m-%d")    # %d.%m.%Y %H:%M:%S.000 GMT%z

# Removing all empty dates
# Build complete timeline from start date to end date
dt_all = pd.date_range(start=df_data[datetime_column].tolist()[0], end=df_data[datetime_column].tolist()[-1])
# Retrieve the dates that ARE in the original dataset
dt_obs = set([d.strftime("%Y-%m-%d") for d in pd.to_datetime(df_data[datetime_column])])
# Define dates with missing values
dt_breaks = [d for d in dt_all.strftime("%Y-%m-%d").tolist() if not d in list(dt_obs)]


df_data_test = df_data[df_data['Date'].dt.year == test_year]
df_data_train = df_data[df_data['Date'].dt.year != test_year]


def trading_plot():
    global counter
    global df_data_train

    if counter < len(df_data_test):
        df_data_train = df_data_train.append(df_data_test.iloc[counter])
        counter += 1
    else:
        df_data_train = df_data

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=[0.7, 0.3],
                        subplot_titles=['OHLC chart', ''])

    # Plot OHLC on 1st subplot
    fig.add_trace(go.Candlestick(x=df_data_train[datetime_column].tolist(),
                                 open=df_data_train["Open"].tolist(), close=df_data_train["Close"].tolist(),
                                 high=df_data_train["High"].tolist(), low=df_data_train["Low"].tolist(),
                                 name=""), row=1, col=1)

    # Plot volume trace on 2nd row
    colors = ['red' if row['Open'] - row['Close'] >= 0 else 'green' for index, row in df_data_train.iterrows()]
    fig.add_trace(go.Bar(x=df_data_train[datetime_column], y=df_data_train['Volume'], name="", marker_color=colors,
                         hovertemplate="%{x}<br>Volume: %{y}"), row=2, col=1)

    # Add chart title and Hide dates with no values and remove rangeslider
    fig.update_layout(title="", height=600, showlegend=False,
                      xaxis_rangeslider_visible=False,
                      xaxis_rangebreaks=[dict(values=dt_breaks)])

    # Update y-axis label
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)

    fig.update_xaxes(showspikes=True, spikecolor="green", spikesnap="cursor", spikemode="across")
    fig.update_yaxes(showspikes=True, spikecolor="orange", spikethickness=2)
    fig.update_layout(spikedistance=1000, hoverdistance=100)

    fig.layout.xaxis.range = ("2022-12-01", "2023-03-01")

    return fig


# The UI of the demo defines here.
with gr.Blocks() as demo:
    gr.Markdown("Auto trade bot.")

    # dt = gr.Textbox(label="Current time")
    # demo.queue().load(get_time, inputs=None, outputs=dt, every=1)

    # for plotly it should follow this: https://gradio.app/plot-component-for-maps/
    candlestick_plot = gr.Plot().style()
    demo.queue().load(trading_plot, [], candlestick_plot, every=1)
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("User Interactive panel.")
            amount = gr.components.Textbox(value="", label="Amount", interactive=True)
            with gr.Row():
                buy_btn = gr.components.Button("Buy", label="Buy", interactive=True, inputs=[amount])
                sell_btn = gr.components.Button("Sell", label="Sell", interactive=True, inputs=[amount])
                hold_btn = gr.components.Button("Hold", label="Hold", interactive=True, inputs=[amount])
        with gr.Column():
            gr.Markdown("Trade bot history.")
            # show trade box history in a table or something
            gr.components.Textbox(value="Some history? Need to decide how to show bot history", label="History", interactive=True)

demo.launch()

