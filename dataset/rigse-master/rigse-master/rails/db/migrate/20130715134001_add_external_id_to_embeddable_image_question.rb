class AddExternalIdToEmbeddableImageQuestion < ActiveRecord::Migration[5.1]
  def change
    add_column :embeddable_image_questions, :external_id, :string
    add_index :embeddable_image_questions, :external_id
  end
end
