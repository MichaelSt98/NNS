# OpenMP introduction

by Intel: 

* [YouTube](https://www.youtube.com/watch?v=nE-xN4Bf8XI&list=PLLX-Q6B8xqZ8n8bwjGdzBJ25X2utwnoEG)
* [GitHub](https://github.com/tgmattso/OpenMP_intro_tutorial)
* [Slides (PDF)](https://github.com/tgmattso/OpenMP_intro_tutorial/blob/master/omp_hands_on.pdf)

## Why parallel programming?

### Software or Hardware responsible for performance?

* until now, Hardware was (mainly) responsable for performance
* End of Moore's law
* Power consumption can be reduced by parallel computing without loosing performance (see [part 1 module 1](https://www.youtube.com/watch?v=cMWGeJyrc9w&list=PLLX-Q6B8xqZ8n8bwjGdzBJ25X2utwnoEG&index=2))
	* **Software needs to be responsable for performance: parallel computing**
	* automatic parallelization is not working! Software developers need to do it by themselves.

See [A. P. Chandrakasan, M. Potkonjak, R. Mehra, J. Rabaey and R. W. Brodersen, "Optimizing power using transformations," in IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems, vol. 14, no. 1, pp. 12-31, Jan. 1995, doi: 10.1109/43.363126.](https://ieeexplore.ieee.org/document/363126)

## Avoid false sharing

See
 
* [06 Discussion 2](https://www.youtube.com/watch?v=OuzYICZUthM&list=PLLX-Q6B8xqZ8n8bwjGdzBJ25X2utwnoEG&index=7) 	 
* [Intel: False sharing](https://software.intel.com/content/www/us/en/develop/articles/avoiding-and-identifying-false-sharing-among-threads.html)