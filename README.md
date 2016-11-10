# bookish-potato
CS4243 Volleyball Project

## Step 1
Move the volleyball videos to a folder called beachVolleyball (pretty much just the folder passed to us in the project zip)

## Step 2
video transformation
`python video_stabilizer.py -v beachVolleyball/beachVolleyball1.mov`

create virtual court
`python virtual_court.py -v beachVolleyball/beachVolleyballX.mov -c hardcoded_court_corners`
