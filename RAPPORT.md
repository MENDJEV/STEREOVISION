//créer et sourcer un environnement virtuel
python3 -m venv venv
source venv/bin/activate

git status //états des fichiers

git checkout .
git pull

git add . //ajout de toutes les fichiers modifoiés et ajoutés
git commit -m "commentaire"  // ajout des commit avec commentaires

git pull --rebase origin main 
git push origin main

git add .gitignore
git commit -m "Remove terraform cache (.terraform) from repository"


git remote -v
