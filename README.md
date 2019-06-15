# Autoencoder Image Denoising

Trabalho de implementação voltado para a disciplina de Redes Neurais (CK0251) da Universidade Federal do Ceará (UFC).  

> Universidade Federal do Ceará  
> Cursos de Graduação em Ciência e Engenharia da Computação  
> Professor: José Maria Monteiro  

### Integrantes
- [Igor Moura](https://github.com/IgorChavesMoura)  
- [Paloma Bispo](https://github.com/PowerPaloma)  
- [Tales Araujo](https://github.com/talesaraujo) 

### Visão Geral
O objetivo deste trabalho é implementar, aplicar e fazer valer-se do uso de uma Rede Neural Profunda (auto-encoder) para a retirada de ruídos (imperfeições) de texto em formato de imagem, de um dataset próprio (KDD).  
Autoencoders (AE) são uma arquitetura de redes neurais artificiais que visam copiar as suas entradas para as suas saídas. Eles fazem a compressão da entrada em uma __representação de espaço latente__ (uma representação simplificada de um espaço dimensional mais complexo), e então reconstroem a saída a partir dessa representação.

<p align="center">
  <img src="https://cdn-images-1.medium.com/max/800/1*V_YtxTFUqDrmmu2JqMZ-rA.png" alt="Autoencoder's architecture image"/>
</p>

A tarefa, em suma, é baseada em duas abstrações:
1. __Encoder__: Esta é a parte da Rede Neural que comprime a entrada em uma representação de espaço latente. Pode ser representado por uma função de codificação \mathit{h = f(x)}
2. __Decoder__: Esta parte visa reconstruir a entrada a partir da representação de espaço latente. Pode ser representada como \mathit{r = g(h)}

Portanto, de maneira complementar, o autoencoder pode ser descrito por uma função $g(f(x))=r$ \mathit{g(f(x)) = r} onde busca-se \mathit{r} o mais próximo possível da entrada \mathit{x}.

### Requisitos
- [Conjunto de bibliotecas scipy](https://www.scipy.org/install.html)
- [Tensorflow](https://www.tensorflow.org/)
- [Keras](https://keras.io/)