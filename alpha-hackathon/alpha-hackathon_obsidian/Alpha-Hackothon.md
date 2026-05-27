работа с object типами данных --> в float/int dtype:
- rate
- eva
- eva_perc
- ncl

**request_id - группируем по нему** 
# Для train/test выборки str -> category

# OE:
- verif_compl
- risk_level_map
- verif_need
- need_2ndfl
# OHE:
- channel
- offer_type
# hz:
- np.log1p(limit/1000000)
- np.log1p(req_loan_amount/1000000)


# Feat:

## удаление (0 uniqua):
- per_capita_income_rur_amt
- lanumberofchildren
- laorganizationtype
- loanapplrealtytype
- loanapplworktype
- lalifeinsurance
- loanapplsocialstatus
- anotherincome_rur_amt
- loanapplpositiontype
- country !!! Можно подумать мб из других стран
- 'gp_date', 'ei_date', 'nb_date' | date колонки