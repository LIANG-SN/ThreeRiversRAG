import torch
from sympy.physics.units import temperature

# Use a pipeline as a high-level helper
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():

    # Load model directly

    # Load the model and tokenizer to the GPU
    if torch.cuda.is_available():
        print("Using GPU")
    model_id = "Qwen/Qwen2-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")

    # Define the pipeline
    pipe = pipeline('text-generation', model=model, tokenizer=tokenizer, torch_dtype=torch.float16)

    intro_info = """You are a smart assistant designed to help come up with reading comprehension questions. You will be given a web-crawled document relevant to topics about Pittsburgh and Carnegie Mellon University (CMU) such as general information/history, events, music, sports, and culture."""
    task = """Based on the document, generate exactly 10 question and answer pairs covering different content topics."""
    requirement = """Each question must be independently answerable and the answers must be directly and exactly found in the document."""
    answer_format = """For each pair, output in this exact format without any extra text:
    Q: "YOUR_QUESTION_HERE", Ans: "YOUR_ANSWER_HERE"."""
    ans_req = """Do not include any introductory text, commentary, or explanations. The final output must contain only 10 Q/A pairs, nothing else. Each answer must be extremely succinct (only key words or phrases) and should not repeat the question."""
    examples = """Examples:
    Q: When was Carnegie Mellon University founded, Ans: 1900"""

    instruct_prompts = [intro_info, task, requirement, ans_req, answer_format, examples]
    INSTRUCTIONS = " ".join(instruct_prompts)
    test_data = """about -     cmu - carnegie mellon university carnegie mellon university — — — search search big ideas start here. cmu ›              about carnegie mellon university challenges the curious and passionate to imagine and deliver work that matters. a private, global research university, carnegie mellon stands among the world's most renowned educational institutions, and sets its own course. start the journey here . over the past 10 years, more than 400 startups linked to cmu have raised more than $7 billion in follow-on funding. those investment numbers are especially high because of the sheer size of pittsburgh’s growing autonomous vehicles cluster – including uber, aurora, waymo and motional – all of which are here because of their strong ties to cmu. with cutting-edge brain science, path-breaking performances, innovative startups, driverless cars, big data, big ambitions, nobel and turing prizes, hands-on learning, and a whole lot of robots, cmu doesn't imagine the future, we create it. about cmu visit admission leadership strategic plan diversity, equity, inclusion and belonging andrew carnegie famously said, "my heart is in the work." at cmu, we think about work a little differently... 14,500+ students representing 100+ countries 1,300+ faculty representing 50+ countries 109,900+ alumni representing 140+ countries view factsheet [pdf] in pittsburgh many seek pittsburgh for being a hot spot for entrepreneurship and a model for future cities . others come for the city's burgeoning food scene . visit us » and around the globe you’ll find cmu locations nationwide — and worldwide. silicon valley. qatar. africa. washington, d.c. to name a few. see global locations & programs » #2 undergraduate computer science u.s. news & world report, 2023 #2 undergraduate research & creative projects u.s. news & world report, 2023 #7 world's best drama schools the hollywood reporter, 2023 #1 graduate information systems u.s. news & world report, 2023 #4 most innovative schools u.s. news & world report, 2023 #15 u.s. universities times higher education, 2023 #24 university in the world times higher education, 2023 more awards & rankings first smile in an email the smiley :-) was created by carnegie mellon research professor scott fahlman on september 19, 1982. this was the beginning of emoticons in email, and the precursor to emojis. ;-)   :-(   :-o fifteen minutes & counting carnegie mellon alumnus andy warhol was an iconic figure in the pop art movement that explored the relationship of art to modern celebrity culture and advertising. his work included hand drawing, painting, printmaking, photography, silk screening, sculpture, film and music. a pioneer in computer-generated art, he is considered one of the most influential artists of the 20th century. changing the way the cookie crumbles it's never too early for women to learn the art of negotiation, and that's why a carnegie mellon professor has partnered with the girl scouts. the first girl scout badge for negotiation, named "win-win: how to get what you want," started with carnegie mellon professor linda babcock. to earn the badge, girls learn why and how negotiation can be useful — and it goes beyond selling cookies. babcock has also co-authored two books on the subject: women don't ask and ask for it: how women can use the power of negotiation to get what they really want . cruise controlled in 1979, carnegie mellon established the nation's first robotics institute. since then, professor and alumnus william "red" whittaker has been a robotics pioneer, founding the discipline of field robotics, developing unmanned vehicles to clean up the three mile island nuclear accident site, and leading the tartan racing team to victory in the $2 million urban challenge robotic autonomous vehicles race. technologies like these can help make driving safer by preventing accidents. little brags of big ideas explore cmu's big ideas — a gallery of innovations and sparks of inspiration that have grown to shape the world. see more brags » seven schools & colleges college of engineering college of fine arts dietrich college humanities and social sciences heinz college information systems & public policy management mellon college of science school of computer science tepper school of business carnegie mellon university challenges the curious and passionate to deliver work that matters. calendar careers covid-19 updates directory / contact feedback global locations health & safety news site map title ix alumni business & research partners faculty & staff students carnegie mellon university 5000 forbes avenue pittsburgh, pa 15213 412-268-2000 legal info www.cmu.edu © 2025 carnegie mellon university cmu on facebook cmu on twitter cmu on linkedin cmu youtube channel cmu on instagram cmu on flickr cmu social media directory academics interdisciplinary programs libraries learning for a lifetime admission undergraduate graduate about leadership vision, mission and values history traditions inclusive excellence pittsburgh rankings awards visit david & susan coulter welcome center maps & getting here research centers & institutes student experience athletics give alumni business & research partners faculty & staff students"""
    input_prompts = INSTRUCTIONS + "\n\n" + "Document content: " + test_data

    with torch.no_grad():
        print("Generating questions...")
        if torch.cuda.is_available():
            print("Using GPU")
            model.to("cuda")
        result = pipe(input_prompts, max_new_tokens=512, temperature = 0.8, top_k = 50, top_p = 0.95)

    print(result[0]["generated_text"][1]["content"])

if __name__ == "__main__":
    main()
