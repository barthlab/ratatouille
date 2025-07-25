import pandas as pd

from kitchen.plotter.color_scheme import TABLEAU_10


def write_boolean_dataframe(df: pd.DataFrame, sheet_name: str, save_path: str, color_second_column: bool = True):
    df_display = df.replace({True: '✅', False: '❌'})
    df_color_column = list(set(df.iloc[:, 1].to_list()))
   
    def set_color(symbol):
        if symbol == '✅':
            return 'color: green'
        elif symbol == '❌':
            return 'color: red'
        elif symbol in df_color_column:
            return f"color: {TABLEAU_10[df_color_column.index(symbol)]}"
        return ''
    
    with pd.ExcelWriter(save_path, engine='xlsxwriter') as writer:
        df_display.style.applymap(lambda _: 'font-weight: bold').applymap(set_color).to_excel(writer, sheet_name=sheet_name, index=False)
        worksheet = writer.sheets[sheet_name]

        worksheet.set_column('A:E', 20)
        worksheet.set_column('F:Z', 10)

    print(f"Saved to {save_path}")