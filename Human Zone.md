Welcome to the human zone, I'm the author of this project (or rather the AI wielder) and the only human being ever participated. I'm here to write down the process of creating this project just in case I'll be using it in the future.

---

Originally this started as I had this long lasting dream of automizing sentence mixing(for those who aren't internet-indulged enough, sentence mixing is basically editing the audio of multiple sentences and creating a new sentence, it entertains people by changing the meanings of the original sentences while keeping the clear signs of editing(the out-of-placeness) visible(otherwise it's gonna smell like counterfeiting and not funny). All you need to know is people (someone out there) believes it may be better than AI at times, and that's why this is here).

---

I designed that the program should do three things: built audio base(you do not want to mess with the source folder) with VAD, ASR the content of it to get the words and timestamps for each before dumping them into a database, and do the sentence mixings using the words and the timestamps, seems simple right?

The early issues were relatively stupid: VAD took a year and won't ASR because I didn't add a minimum length to the audio files in the audio base. Back then I made it that the audio inputs are stored in multiple labeled files, and the VAD happily created more than 10,000 files out of six hours' audio, with many less than a second and causing a rare CUDA exception that the AI had no clue about fixing. I fixed it by adding a 20s threshold, making sure each audio file consists of integer amount of sentences and longer than 20s. The runtime instantly fell by more than 90% as all those small IOs wreaked havoc upon my SSD. Now I switched to one file base, so as to save a column in the database(does not actually there's still data there hasn't removed it yet)

All above were pretty much done in the first day. Then I had to battle with pause and resume after restarting the backend(seems so unimportant right now) before coming to realize the whole project doesn't work. Whisper is quite an old and classic model for ASR and it did it fine, it's the timestamps that were messed up. Only fragments of words got pieced up together and none of the sentences produced contains a single full word. Turns out whisper is good enough for aligning subtitles but not enough for sentence mixing.

So I had to come up with ways to improve the accuracy of them timestamps. I tried using VAD (telling AI to use it anyways) but probably MFA saved the day. It's a little hard to deploy considering this software aims to help with relatively lazy people(me).

Oh and during the writing of this, I came up with the ultimate solution: human. This is way smarter than figuring out how to make progress bars without costing too much. Now there will be a mode that throws all the audio clips to humans for editing. This works so people no longer have to search those words in countless audio files and clip them out. All they need to do is dump the clips into the AU and cut off those unwanted bits.