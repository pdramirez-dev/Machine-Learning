
import entrenar as en

loop = True
while loop:
    print("########## ---Selecciones la Opcion")
    print('1-)Entrenar el Modelo')
    print('2-)Provar el Modelo')
    print('0-) Salir Programa')
    val = input()
    val = int(val)
    if val == 1:
        en.fit_model()
        en.save_model("")
