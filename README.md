# feel-smart-llm

Everyone is using AI right now. Prompting it, building with it, arguing 
about it on LinkedIn. But almost nobody understands what is actually 
happening underneath. So I built one from scratch to show you exactly 
how it works.

No cloud. No API. No PyTorch. Just Python, numpy, a couple hundred 
lines of code, and one paragraph of text.

## What is an LLM

LLM stands for Large Language Model. It is software trained on massive 
amounts of text from the internet, books, articles, code, conversations, 
and probably your tweets from 2014 that you forgot about.

Here is the part that matters. It does not think. It predicts. Given 
what you just said, it calculates the most likely next word, then the 
next, then the next. It does this so well that it feels like a 
conversation but underneath it is just math.

## How to run it

pip3 install numpy
python3 llm.py "your starting text here"

That is it. No other dependencies.

## What happened when I ran it

First run. 400 epochs. Loss dropped from 3.45 to 2.07.

Seeded with "It does not think":

    It does not thinker f LLLMelofs pin ats pss Ims iverers sxt...

Beautiful right. Gibberish but interesting gibberish. It found LLLM. 
It has fragments like "at" and "It." It is learning the shape of 
English. It just has no idea what any of it means yet.

## Then it blew up

Changed epochs to 2000 and ran it again. Around epoch 1280:

    RuntimeWarning: overflow encountered in matmul
    Epoch 1360  |  loss = nan
    Epoch 1440  |  loss = nan
    ValueError: probabilities contain NaN

NaN means Not a Number. The math broke. The model did not just stop 
improving, it destroyed itself. This is called exploding gradients. 
Every major lab has dealt with this. OpenAI, Google, Meta, all of them. 
They just have fancier tools to catch it before it happens. Same crash, 
different budget.

## The fix was two lines of code

Lower the learning rate from 0.05 to 0.01. Then add this:

    for key in grads:
        np.clip(grads[key], -1.0, 1.0, out=grads[key])

That puts a hard ceiling on how large any single weight update can be. 
PyTorch does this automatically behind the scenes. When you build from 
scratch you have to do it yourself which is annoying but it also means 
you actually understand why it exists.

## After the fix. 4000 epochs.

    Epoch    0  |  loss = 3.49
    Epoch  400  |  loss = 2.92
    Epoch 1400  |  loss = 2.69
    Epoch 4000  |  loss = 2.17

Seeded with "It does not think":

    It does not thinkt. that, ad. boe mes tiy. pperssheonnggennext 
    lxte pat La jod cuigeanes...

Still not Shakespeare. But look. "that." "next." "why." "LLM." Real 
words climbing out of the chaos.

## The actual lesson

I could push this to 10,000 epochs. It would not matter much. The 
model has memorized everything it can from 535 characters. It is not 
a training problem anymore. It is a data problem.

More data beats more epochs every single time. This is the same lesson 
the entire industry learned the hard way between 2018 and 2022. Everyone 
kept tuning learning rates and training longer. Then someone realized 
the breakthrough was not training harder. It was feeding the model 
more data.

## The thing that got me

This model has about 10,000 parameters. GPT-4 has a rumored 1.7 
trillion. The architecture is the same. The math is the same. The 
problems are the same. I hit the same wall they hit. I used the same 
fix they use. The only difference is they have more data and more 
compute. That is it. There is no secret sauce. There is no magic. 
It is prediction at scale.

## How to make it better

Replace the TEXT variable at the top with anything longer. A full 
article. A Wikipedia page. A few pages of a book. The model needs 
new patterns to learn from, not more repetitions of the same ones.
