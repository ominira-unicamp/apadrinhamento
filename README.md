# **Plataforma de Apadrinhamento CTComp**

A plataforma de apadrinhamento utiliza embeddings da OpenAI para algoritmicamente fazer um match entre melhores padrinhos e calouros que possuem compatibilidade segundo pesos definidos em múltiplas categorias. Atualmente o algoritmo de match está implementado, posteriormente será criada uma API em volta do algoritmo, bem como um site para inscrição de calouros e veteranos e moderação do processo de match.

## Como Rodar?

- Crie um Ambiente Virutal (venv)
  * ```python3 -m venv ./venv```
  * ```source ./venv/bin/activate```
- Vá para a pasta do backend e instale os requisitos
  * ```cd backend```
  * ```pip install -r requirements.txt```
- Utilize o .env.template para criar um .env
- Preencha as informações no .env de acordo com as suas
- Rode o código com ```python3 match.py```
