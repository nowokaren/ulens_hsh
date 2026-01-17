ULTIMA ACTUALIZACIÓN - 22/08/2020

reduc.py (reduc_vp.py) - Programa de reducción de imágenes astronómicas

Tareas: 
- Combinar bias para generar un masterbias.
- Combinar flats para generar masterflats en diferentes bandas.
- Calibrar imágenes de ciencia.
- Resolver astrometría usando astrometry.net

Importante:
En laruidosa hay que activar el entorno de python3.

Antes de correr el programa, el usuario deberá indicar que tipo de calibraciones desea realizar. Ingresando al archivo reduc.py, en las líneas 25 a 28, encontrará cuatro variables lógicas: zerocorrection, flatcorrection, astrometry y rm_images. Cuando estas variables son verdaderas (True), se le permite al script realizar esa etapa de reducción.
Esto podría permitir al usuario, por ej., hacer la calibración de todas las imágenes por bias, sin necesidad de realizar las posteriores calibraciones. Sin embargo, la reducción tiene un orden lógico y no permite calibrar imágenes por flat si antes no se les ha restado el bias. Lo mismo para con la parte astrométrica. Para acceder a esto, las imágenes primero deben haber sido calibradas por bias y flats.
En el caso que no haya imágenes para corregir por bias/flat, el programa deja de correr con un mensaje de error.
En la versión "reduc_vp.py", se le dice que tipo de calibraciones realizar ingresando el valor de las variables lógicas por pantalla.

Una vez que todas las calibraciones terminan, se eliminan todas las imágenes que no sirven si la variable rm_images es True. Esto permite poder hacer pruebas isn necesidad de volver a corregir las imágenes por bias y/o flats. Una vez que todas las pruebas hayan terminado, se puede volver a correr el script con todas las variables en False, excepto rm_images. De esta forma elimina las imágenes que sobran. Sólo se guardan las originales, los flats corregidos por bias, el masterbias, los marterflats y las '_wcs.fits'.

Cada vez que el programa corra de nuevo se creará el masterbias y los masterflats (si sus variables lógicas son verdaderas). La astrometría no correrá de nuevo ya que esta parte tarda mucho. Para correr nuevamente la astrometría de una imagen primero hay que eliminar la '_wcs.fits' de esa imagen y volver a correr. Si la versión de esa imagen corregida por bias y flat ha sido eliminada, es necesario que las variables zerocorrection y flatcorrection sean True.

Requerimientos:
- Python 3
- astropy: astropy is installed by default with the Anaconda Distribution. To update to the latest version run  "conda update astropy"
- ccdproc: "conda install -c astropy ccdproc"
- pathlib: "conda install -c menpo pathlib"
- multiprocessing
- time
- astrometry/sextractor: esto si no se como se instala. OJO: en la parte del programa donde se resuelve la astrometría hay que poner el path donde se encuentra sextractor instalado en cada una de sus computadoras.

