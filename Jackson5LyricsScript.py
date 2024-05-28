import os
import lyricsgenius

# Create a Genius API object
genius = lyricsgenius.Genius("KJ8LWtb5l21IkZIIYoeXRqnUr4Vq6-HLOWylr7h9TSWw6UkhuorAPP0f2aQzAxND")

# Define a function to clean the lyrics
def clean_lyrics(lyrics):
    # Split the lyrics by line breaks
    lines = lyrics.split('\n')
    cleaned_lines = []
    for line in lines:
        # Exclude the first line (title)
        if line != lines[0]:
            # Exclude lines containing certain tags
            if not any(tag in line.lower() for tag in ["[chorus]", "[intro]", "[verse", "[outro]" , "[bridge]"]):
                cleaned_lines.append(line)
    # Join the cleaned lines back together
    cleaned_lyrics = '\n'.join(cleaned_lines)
    return cleaned_lyrics.strip()

# Define the tracklists with song titles for each album
album_tracklists = {
    "Diana Ross Presents The Jackson 5": [
    "Zip A Dee Doo Dah",
    "Nobody",
    "I Want You Back",
    "Can You Remember",
    "Standing In The Shadows Of Love",
    "You've Changed",
    "My Cherie Amour",
    "Who's Loving You",
    "(I Know) I'm Losing You",
    "Chained",
    "Stand",
    "Born To Love You"
],
    "ABC": ["The Love You Save", "One More Chance", "ABC", "2-4-6-8", "(Come 'Round Here) I'm the One You Need", "Don't Know Why I Love You", 
            "Never Had a Dream Come True", "True Love Can Be Beautiful", "La-La (Means I Love You)", "I'll Bet You", "I Found That Girl", 
            "The Young Folks"],
    "Third Album": ["I'll Be There", "Ready or Not (Here I Come)", "Oh How Happy", "Bridge Over Troubled Water", "Can I See You in the Morning", 
"Goin' Back to Indiana", "How Funky Is Your Chicken", "Mama's Pearl", "Reach In", "The Love I Saw in You Was Just a Mirage", "Darling Dear"],
 "Maybe Tomorrow": [
        "Maybe Tomorrow",
        "She's Good",
        "Never Can Say Goodbye",
        "The Wall",
        "Petals",
        "Sixteen Candles",
        "(We've Got) Blue Skies",
        "My Little Baby",
        "It's Great to Be Here",
        "Honey Chile",
        "I Will Find a Way"
    ],
    "Goin' Back to Indiana": [
        "I Want You Back",
        "Maybe Tomorrow",
        "The Day Basketball Was Saved",
        "Stand",
        "I Want to Take You Higher",
        "Feelin' Alright",
        "Walk On",
        "The Love You Save",
        "Goin' Back to Indiana"
    ],
    "Lookin' Through the Windows": [
        "Ain't Nothing Like the Real Thing",
        "Lookin' Through the Windows",
        "Don't Let Your Baby Catch You",
        "To Know",
        "Doctor My Eyes",
        "Little Bitty Pretty One",
        "E-Ne-Me-Ne-Mi-Ne-Moe (The Choice Is Yours to Pull)",
        "If I Have to Move a Mountain",
        "Don't Want to See Tomorrow",
        "Children of the Light",
        "I Can Only Give You Love"
    ],
    "Skywriter": [
        "Skywriter",
        "Hallelujah Day",
        "The Boogie Man",
        "Touch",
        "Corner of the Sky",
        "I Can't Quit Your Love",
        "Uppermost",
        "World of Sunshine",
        "Ooh, I'd Love to Be With You",
        "You Made Me What I Am",
        "Get It Together"
    ],
    "G.I.T.: Get It Together": [
        "Get It Together",
        "Don't Say Nothin' Bad (About My Baby)",
        "Ready or Not (Here I Come)",
        "You Need Love Like I Do (Don't You?)",
        "Mama I Gotta Brand New Thing (Don't Say No)",
        "It's Too Late to Change the Time",
        "You've Got a Friend"
    ],
    "Dancing Machine": [
        "I Am Love",
        "Whatever You Got, I Want",
        "She's a Rhythm Child",
        "Dancing Machine",
        "The Life of the Party",
        "What You Don't Know",
        "If I Don't Love You This Way",
        "It All Begins and Ends with Love",
        "The Mirrors of My Mind"
    ]
}

# Create a directory to store the albums' folders
if not os.path.exists("Jackson_Five"):
    os.makedirs("Jackson_Five")

# Save lyrics of each song in a text file within the corresponding album's folder
for album_name, tracklist in album_tracklists.items():
    # Create a folder for the album
    album_folder = f"Jackson_Five/{album_name}"
    if not os.path.exists(album_folder):
        os.makedirs(album_folder)
    
    # Save lyrics of each song in a text file
    for song_title in tracklist:
        # Search for the song lyrics
        song = genius.search_song(song_title, "Jackson Five")
        if song:
            lyrics = clean_lyrics(song.lyrics)
            with open(f"{album_folder}/{song_title}.txt", "w", encoding="utf-8") as file:
                file.write(lyrics)
        else:
            print(f"Lyrics not found for {song_title}")

print("Lyrics saved successfully!")
