use File::Basename;
use File::Glob ':glob';
use File::Path qw(remove_tree);

# directories
my $dir_images = "data/images";
my $dir_annotations = "data/annotations";

my $removed_annotations = 0;
my $removed_files = 0;
# first, remove all image files that do not have a corresponding annotation file
my @img_files = bsd_glob("$dir_images/*");
my %file_list_map = map {fileparse($_) => 1} @img_files;
foreach my $img_file (@img_files) {
    my $base_file = fileparse($img_file, qr/\.[^.]*/);
    my $anno_file = "$dir_annotations/$base_file.json";
    if (!-e $anno_file) {
         unlink($img_file);
         $removed_annotations++;
         delete $file_list_map{$base_file};
    }
}

# calculate 95% of the total files
my $num_files = scalar keys(%file_list_map);
my $remove_files = int($num_files * 95 / 100);
my @file_list = keys %file_list_map;
for my $i (0..$remove_files) {
    # select a random file
    my $rand_file = splice @file_list, (rand @file_list), 1;
    my $base_file = fileparse($rand_file);
    my $img_file = "$dir_images/$base_file";
    my $anno_file = "$dir_annotations/" . fileparse($base_file, qr/\.[^.]*/) . ".json";

    # if the file exists in images and annotations directory, remove it
    if (-e $img_file && -e $anno_file) {
        unlink($img_file);
        unlink($anno_file);
        $removed_files++;
    }
}

print("Removed $removed_annotations files due to missing annotation file\n");
print("Removed $removed_files files\n");