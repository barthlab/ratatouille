import pandas as pd


def write_boolean_dataframe(df: pd.DataFrame, sheet_name: str, save_path: str):
    df_display = df.replace({True: '✅', False: '❌'})
    def set_color(symbol):
        if symbol == '✅':
            return 'color: green'
        elif symbol == '❌':
            return 'color: red'
        return ''
   
    with pd.ExcelWriter(save_path, engine='xlsxwriter') as writer:
        df_display.style.applymap(set_color).to_excel(writer, sheet_name=sheet_name, index=False)
        worksheet = writer.sheets[sheet_name]

        worksheet.set_column('A:D', 20)
        worksheet.set_column('E:Z', 10) 

    print(f"Saved to {save_path}")