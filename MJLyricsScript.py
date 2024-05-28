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
    "Got to Be There": ["Ain't No Sunshine", "I Wanna Be Where You Are", "Girl Don't Take Your Love from Me", "In Our Small Way", "Got to Be There", 
                        "Rockin' Robin", "Wings of My Love", "Maria (You Were the Only One)", "Love Is Here and Now You're Gone", "You've Got a Friend"],
    "Ben": ["Ben", "Greatest Show on Earth", "People Make the World Go Round", "We've Got a Good Thing Going", "Everybody's Somebody's Fool",
            "My Girl", "What Goes Around Comes Around", "In Our Small Way", "Shoo-Be-Doo-Be-Doo-Da-Day", "You Can Cry on My Shoulder"],
    "Music & Me": ["With a Child's Heart", "Up Again", "All the Things You Are", "Happy", "Too Young", "Doggin' Around", "Johnny Raven",
                    "Euphoria", "Morning Glow", "Music and Me"],
    "Forever, Michael": ["We're Almost There", "Take Me Back", "One Day in Your Life", "Cinderella Stay Awhile", "We've Got Forever",
                            "Just a Little Bit of You", "You Are There", "Dapper Dan", "Dear Michael", "I'll Come Home to You"],
    "Off the Wall": ["Don't Stop 'Til You Get Enough", "Rock with You", "Working Day and Night", "Get on the Floor", "Off the Wall", 
                        "Girlfriend", "She's Out of My Life", "I Can't Help It", "It's the Falling in Love", "Burn This Disco Out"],
    "Thriller": ["Wanna Be Startin' Somethin'", "Baby Be Mine", "The Girl Is Mine (with Paul McCartney)", "Thriller", "Beat It", 
                    "Billie Jean", "Human Nature", "P.Y.T. (Pretty Young Thing)", "The Lady in My Life"],
    "Bad": ["Bad", "The Way You Make Me Feel", "Speed Demon", "Liberian Girl", "Just Good Friends (feat. Stevie Wonder)", "Another Part of Me",
                "Man in the Mirror", "I Just Can't Stop Loving You (with Siedah Garrett)", "Dirty Diana", "Smooth Criminal", "Leave Me Alone"],
    "Dangerous": ["Jam", "Why You Wanna Trip on Me", "In the Closet", "She Drives Me Wild", "Remember the Time", "Can't Let Her Get Away", 
                    "Heal the World", "Black or White", "Who Is It", "Give in to Me", "Will You Be There", "Keep the Faith", 
                    "Gone Too Soon", "Dangerous"],
    "HIStory- Past, Present and Future, Book I": ["Scream (with Janet Jackson)", "They Don't Care About Us", "Stranger in Moscow", 
                                                    "This Time Around (feat. The Notorious B.I.G.)", "Earth Song", "D.S.", "Money", 
                                                    "Come Together", "You Are Not Alone", "Childhood (Theme from Free Willy 2)", 
                                                    "Tabloid Junkie", "2 Bad", "HIStory", "Little Susie", "Smile"],
    "Blood on the Dance Floor- HIStory in the Mix": ["Blood on the Dance Floor", "Morphine", "Superfly Sister", "Ghosts", "Is It Scary", 
                                                        "Scream Louder (Flyte Tyme Remix) (with Janet Jackson)", "Money (Fire Island Radio Edit)", 
                                                        "2 Bad (Refugee Camp Mix)", "Stranger in Moscow (Tee's In-House Club Mix)", 
                                                        "This Time Around (D.M. Radio Mix) (feat. The Notorious B.I.G.)", 
                                                        "Earth Song (Hani's Club Experience)", "You Are Not Alone (Classic Club Mix)"],
    "Invincible": ["Unbreakable", "Heartbreaker", "Invincible", "Break of Dawn", "Heaven Can Wait", "You Rock My World", "Butterflies", 
                        "Speechless", "2000 Watts", "You Are My Life", "Privacy", "Don't Walk Away", "Cry", "The Lost Children", 
                        "Whatever Happens", "Threatened"]
}

# Create a directory to store the albums' folders
if not os.path.exists("Michael_Jackson"):
    os.makedirs("Michael_Jackson")

# Save lyrics of each song in a text file within the corresponding album's folder
for album_name, tracklist in album_tracklists.items():
    # Create a folder for the album
    album_folder = f"Michael_Jackson/{album_name}"
    if not os.path.exists(album_folder):
        os.makedirs(album_folder)
    
    # Save lyrics of each song in a text file
    for song_title in tracklist:
        # Search for the song lyrics
        song = genius.search_song(song_title, "Michael Jackson")
        if song:
            lyrics = clean_lyrics(song.lyrics)
            with open(f"{album_folder}/{song_title}.txt", "w", encoding="utf-8") as file:
                file.write(lyrics)
        else:
            print(f"Lyrics not found for {song_title}")

print("Lyrics saved successfully!")
