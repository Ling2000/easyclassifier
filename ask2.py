from gradio_client import Client
import json 


client = Client("https://huggingface-projects-llama-2-7b-chat.hf.space/--replicas/gm5p8/")



text_path='cor.json'
out_path='ans1.json'
ctxlen=1500

f= open(text_path)
data=json.load(f)

##syspro="Below are a series of dialogues between IMBD and an AI classifier. The AI tries to be helpful, polite, honest, sophisticated, emotionally aware, and humble-but-knowledgeable. The AI will read a JSON file which include some information about the movie. AI will base on the information inside the JSON to classify the movie into the right genre. It also tries to avoid giving false or misleading information, and it caveats when it isn't entirely sure about the right answer. That said, the assistant is practical and really does its best, and doesn't let caution get too much in the way of being useful."
syspro="Below are a series of dialogues between various people and an AI assistant. The AI tries to be helpful, polite, honest, sophisticated, emotionally aware, and humble-but-knowledgeable. The assistant is happy to help with almost anything, and will do its best to understand exactly what is needed. It also tries to avoid giving false or misleading information, and it caveats when it isn't entirely sure about the right answer. That said, the assistant is practical and really does its best, and doesn't let caution get too much in the way of being useful. AI would like to choose one of the following words as the genre of the given JSON:Action, Adventure, Animation, Biography, Comedy, Crime, Documentary, Drama, Family, Fantasy, Film-Noir, Game-Show, History, Horro, Music, Musical, Mystery, News, Reality-TV, Romance, Sci-Fi, Short, Sport, Talk-Show, Thriller, War, Western. All the answer should following the follow format, Reasoning base on the information provided, change to another line, start with a star and answer the genre directly, change line."
correct_prompt = """\
[INST] empty is not the best genre to describe the movie in the JSON. Fill the best matching movie genre for this json, {"movie_id": "tt0144618", "movie_title": "The Spirit of Christmas", "movie_poster": "https://m.media-amazon.com/images/M/MV5BYzgxYWY2ZGMtZDFiYy00OGM4LWI3ODctZGIwMGYwYzM3OTdiXkEyXkFqcGdeQXVyNDE4OTY5NzI@.jpg", "movie_storyLines": ["Four children, all but one of whom go unnamed, build a snowman which comes to life and threatens their town. Kenny, the only child whose name is given in the film, and who resembles the character called Cartman in the subsequent Spirit of Christmas, The (1995) and \"South Park\" (1997), is immediately killed by the monster. When the children go to Santa for help, he is revealed to be a monster in disguise, and slaughters another one of the kids. The two survivors approach the baby Jesus, who takes on the evil snowman and wins, saving Christmas."], "movie_genre": []}. Only answer the movie genre. [/INST]
Reasoning based on the information provided: The Spirit of Christmas is a fantasy movie, as it features supernatural elements such as a snowman coming to life and a monster disguising itself as Santa Claus. The movie's plot revolves around the threat of the monster to a small town and the children's quest to seek help from Santa Claus, who is revealed to be a monster in disguise. The use of magical and supernatural elements, as well as the fantastical setting of a small town during the Christmas season, further classify this movie as a fantasy film.
* Fantasy
I hope this helps! Let me know if you have any other questions.
"""

with open(out_path, 'w') as lab_file:
	molist=list()
	for i in range(len(data)):
		tpl=" Fill the movie genre in this json,"
		mov=data[i]
		if mov["movie_genre"]=="":
			mov["movie_genre"]='empty'
		tpl=mov["movie_genre"]+" may not be the best genre to describe the movie in the JSON. Fill the best matching movie genre for this json, "
		#mov["movie_storyLines"]=" ".join(mov["movie_storyLines"])
		#print(len(mov["movie_storyLines"]))
		mov["movie_storyLines"]=mov["movie_storyLines"][0:3000]
		mov["movie_genre"]=list()
		movie=json.dumps(mov)
		tpl=tpl+movie+". Only answer the movie genre."
		#print(tpl)
		try:
			result = client.predict(
					correct_prompt+tpl,	# str  in 'Message' Textbox component
					syspro,	# str  in 'System prompt' Textbox component
					800,	# int | float (numeric value between 1 and 2048) in 'Max new tokens' Slider component
					0.1,	# int | float (numeric value between 0.1 and 4.0) in 'Temperature' Slider component
					1,	# int | float (numeric value between 0.05 and 1.0) in 'Top-p (nucleus sampling)' Slider component
					1,	# int | float (numeric value between 1 and 1000) in 'Top-k' Slider component
					1,	# int | float (numeric value between 1.0 and 2.0) in 'Repetition penalty' Slider component
					api_name="/chat"
			)
			
		except:
			print('f')
			break
		result=result.split('\n')
		print(i)
		print(result)
		mov["movie_genre"]=result[1][2:]
		molist.append(mov)
		client = Client("https://huggingface-projects-llama-2-7b-chat.hf.space/--replicas/gm5p8/")
	movie=json.dumps(molist)
	lab_file.write(movie+'\n')
