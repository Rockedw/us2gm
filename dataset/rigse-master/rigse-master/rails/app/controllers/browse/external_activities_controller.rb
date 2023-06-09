class Browse::ExternalActivitiesController < ApplicationController
  include Materials::DataHelpers

  # GET /browse/external_activities/1
  def show
    @back_to_search_url = request.env["HTTP_REFERER"]
    if @back_to_search_url && !@back_to_search_url.include?(search_url)
      @back_to_search_url = nil
    end
    if request.post?
      @back_to_search_url = url_for :controller => '/search', :action => 'index',:search_term=>params["search_term"],:activity_page=>params["activity_page"],:investigation_page=>params["investigation_page"]
    end

    material = ::ExternalActivity.find(params[:id])

    @search_material = Search::SearchMaterial.new(material, current_visitor)
    @search_material.url = url_for(@search_material.url)
    @search_material.parent_material.url = url_for(@search_material.parent_material.url)

    @material_data = materials_data([material], nil, 0).shift()

    page_meta = @search_material.get_page_title_and_meta_tags
    @page_title = page_meta[:title]
    @meta_tags = page_meta[:meta_tags]
    @open_graph = page_meta[:open_graph]

  end

end
